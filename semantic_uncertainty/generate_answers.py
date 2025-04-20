"""Sample answers from LLMs on QA task.
This script generates answers using language models and computes various metrics
including accuracy and uncertainty measures."""

# Standard library imports
import gc
import os
import logging
import random
from tqdm import tqdm

# Scientific computing and ML imports
import numpy as np
import torch
import wandb

# Custom module imports
from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute
from init import initialize_huggingface

# Setup logging
utils.setup_logger()

#initialize huggingface
initialize_huggingface()

def main(args):
    """Main function to generate answers and compute metrics.

    Args:
        args: Configuration arguments including model settings, dataset choices,
              and evaluation parameters.
    """
    # Handle dataset-specific configurations
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    # Initialize experiment tracking and setup
    experiment_details = {'args': args}
    random.seed(args.random_seed)
    user = os.environ.get('USER', 'colab_user')  # Use 'colab_user' as a default value
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')

    # Create uncertainty directory if it doesn't exist
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    # Initialize wandb for experiment tracking
    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric based on configuration
    metric = utils.get_metric(args.metric)

    # Load and prepare datasets
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)

    # Handle out-of-distribution training dataset if specified
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Split dataset into answerable and unanswerable questions
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    # Handle answerable-only configuration
    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    # Create few-shot prompt for model
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Setup prompt generation
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    # start del
    logging.info("**"*80)
    logging.info("construct_fewshot_prompt-from_indices calling :")
    logging.info("train_dataset")
    logging.info(train_dataset)
    logging.info("prompt_indices")
    logging.info(prompt_indices)
    logging.info("BRIEF")
    logging.info(BRIEF)
    logging.info("arg")
    logging.info(arg)
    logging.info("**" * 80)
    #end del
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize the language model
    model = utils.init_model(args)

    # Setup p_true baseline if requested
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt, num_generations=args.num_generations,
            metric=metric)
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt

    # Start answer generation
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')

    # Process both train and validation splits
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # Initialize storage for results
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        # Select appropriate dataset and indices based on split
        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))
        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Sample subset of dataset for evaluation
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        # Generate answers for each example
        it = 0
        for index in tqdm(indices):
            # Periodic garbage collection
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # Get example and prepare input
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []

            # Determine number of generations needed
            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            # Generate multiple answers with different temperatures
            for i in range(num_generations):
                # First generation uses low temperature (0.1), others use specified temperature
                temperature = 0.1 if i == 0 else args.temperature

                predicted_answer, token_log_likelihoods, embedding = model.predict(
                    local_prompt, temperature)
                embedding = embedding.cpu() if embedding is not None else None

                # Compute accuracy if needed
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0

                # Handle first (low temperature) prediction
                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    if args.use_context:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc}
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example)})
                else:
                    # Store additional diverse predictions
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            # Store all predictions for this example
            generations[example['id']]['responses'] = full_responses

            # Compute p_true if requested (validation only)
            if args.compute_p_true and dataset_split == 'validation':
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)

        # Save results for current split
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        # Save uncertainty measures for validation split
        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            utils.save(results_dict, 'uncertainty_measures.pkl')

    # Save final experiment details and cleanup
    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model

if __name__ == '__main__':
    # Parse arguments and run main functions
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # Generate answers
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    # Compute uncertainty measures if requested
    if args.compute_uncertainties:
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')