# This code computes various uncertainty measures after generating answers in a text-generation or question-answering context.
# It loads previously generated answers, calculates different uncertainty metrics (such as predictive entropy, semantic entropy,
# p_ik, etc.), and logs these metrics using Weights & Biases (wandb). It also optionally deploys an entailment model for computing
# semantic clusters or verifying if a context entails a response.

from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb

# This import presumably contains analysis functions for the project.
from analyze_results import analyze_run

# This loads the dataset in a particular format.
from uncertainty.data.data_utils import load_ds

# This imports a function for training a classifier that distinguishes correct from incorrect answers (p_ik).
from uncertainty.uncertainty_measures.p_ik import get_p_ik

# Functions for computing semantic entropy and related measures.
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response

# Different classes for entailment checking with various models (Deberta, GPT-4, GPT-3.5, GPT-4-Turbo, Llama, etc.).
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama

# Utility to compute p_true (the probability that a generated response is *actually* correct or true).
from uncertainty.uncertainty_measures import p_true as p_true_utils

# Project-specific utilities (e.g., logger setup, argument parser, model init, etc.).
from uncertainty.utils import utils


# Setup the project's logger at the start.
utils.setup_logger()

# Name of a pickled file storing experiment details.
EXP_DETAILS = 'experiment_details.pkl'


def main(args):
    """
    Orchestrates the process of computing or recomputing various uncertainty measures
    (predictive entropy, p_ik, p_true, etc.). Also handles model restoration from wandb runs
    and optional distribution shift scenarios where training and evaluation data differ.
    """

    # If train_wandb_runid is not specified, reuse the evaluation wandb run ID for training.
    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    # Use environment variables to build paths for Weights & Biases (wandb).
    user = os.environ.get('USER', 'colab_user')  # Use 'colab_user' as a default value
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)

    # Choose the project name based on whether we're in debug mode or not.
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"

    # If new wandb run ID is requested, we create one from an old run's config; else reuse active wandb ID.
    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
            # Merge relevant config from old run with the new script arguments.
            config={**old_run.config, **args.__dict__},
        )

        # This helper function restores artifacts (files) from the old run into our local wandb.run.dir.
        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
    else:
        logging.info('Reuse active wandb id.')

        # If reusing the current active wandb run, we simply build a file path inside wandb.run.dir.
        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    # Determine if we are evaluating on a distribution-shift dataset (train != eval).
    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)
        is_ood_eval = True
        api = wandb.Api()

        # Download the training generations from a separate wandb run if there's distribution shift.
        old_run_train = api.run(f'{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}')
        filename = 'train_generations.pkl'
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)

        # Optionally update the wandb config with the dataset used for training, labeling it as OOD.
        wandb.config.update(
            {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False
        # If there's no distribution shift, and we need p_ik, we load the training data from the same run.
        if args.compute_p_ik or args.compute_p_ik_answerable:
            train_generations_pickle = restore('train_generations.pkl')
            with open(train_generations_pickle.name, 'rb') as infile:
                train_generations = pickle.load(infile)

    # Log and update wandb config to indicate whether or not this is an OOD evaluation.
    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load an entailment model if we plan on computing predictive entropy or other semantic measures.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError("Unsupported entailment model selected.")
        logging.info('Entailment model loading complete.')

    # Optionally compute p_true in this script (less common usage).
    if args.compute_p_true_in_compute_stage:
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        # If we reuse the same entailment model for p_true, that's handled here.
        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        # Load the training dataset used to compute p_true. (p_true is the probability that an answer is indeed correct.)
        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)
        del pt_validation_dataset

        # Possibly limit the number of generations used for p_true to a specific amount.
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError("If not using all generations, must specify a valid number of generations.")
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        # Construct few-shot prompts and gather responses to compute p_true across the dataset.
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric(old_exp['args'].metric))
        del p_true_responses
        # del start
        example = pt_train_dataset[0]
        logging.info("**" * 80)
        logging.info("p_true_few_shot_prompt inputs")
        logging.info("model:")
        logging.info(pt_model)
        logging.info("pt_train_dataset:")
        logging.info(pt_train_dataset)
        logging.info("pt_train_dataset['id']:")
        logging.info(example['id'])
        logging.info("pt_train_dataset['question']:")
        logging.info(example['question'])
        logging.info("pt_train_dataset['context']:")
        logging.info(example['context'])
        logging.info("pt_train_dataset['answers]:")
        logging.info(example['answers'])
        logging.info("old_exp['p_true_indices']:")
        logging.info(old_exp['p_true_indices'])
        logging.info("old_exp['prompt']")
        logging.info(old_exp['prompt'])
        logging.info("old_exp['BRIEF']")
        logging.info(old_exp['BRIEF'])
        logging.info("brief_always:")
        logging.info(old_exp['args'].brief_always and old_exp['args'].enable_brief)
        logging.info("make_prompt")
        logging.info(utils.get_make_prompt(old_exp['args']))
        logging.info("num_generations")
        logging.info(num_gen)
        logging.info("metric")
        logging.info(utils.get_metric(old_exp['args'].metric))
        logging.info("**" * 80)
        # del end
        # Log how many data points we used for p_true few-shot examples.
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    # If recomputing accuracy is requested, we set up the relevant metric from the utils.
    if args.recompute_accuracy:
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = utils.get_metric(args.metric)

    # Restore the previously computed results (from generate_answers.py) that we want to extend or update.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)

    # We store semantic_ids for each validation datapoint inside result_dict.
    result_dict['semantic_ids'] = []

    # Load the validation generations from a pickled file, which includes questions, contexts, generated responses, etc.
    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    # We'll store computed measures in `entropies`, keyed by measure name.
    entropies = defaultdict(list)

    # We'll keep track of each validation example's embedding (from the best or "most likely" answer),
    # whether it's objectively correct, and whether it's answerable.
    validation_embeddings, validation_is_true, validation_answerable = [], [], []

    # We'll store p_true values if requested.
    p_trues = []

    count = 0

    # Helper function to determine if the data point had an actual reference answer or if it's unanswerable.
    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # Main loop over each validation example to compute or recompute uncertainty measures.
    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        question = example['question']
        context = example['context']

        # full_responses is a list of (response_text, token_log_likelihoods) for each generation.
        full_responses = example["responses"]
        logging.info('the response is  %s', full_responses)
        most_likely_answer = example['most_likely_answer']

        # If limiting the number of generations, slice them; otherwise use all.
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError("You must specify a number of generations if not using all.")
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]
        #del start
        logging.info("**"*80)
        logging.info("expanding example")
        logging.info("example")
        logging.info(example)
        logging.info("question")
        logging.info(question)
        logging.info("context")
        logging.info(context)
        logging.info("most_likely_answer")
        logging.info(most_likely_answer)
        logging.info("responses")
        logging.info(responses)
        logging.info("**"*80)
        #del end
        # Optionally recompute accuracy for each most_likely_answer if the user asked for it.
        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                acc = metric(most_likely_answer['response'], example, None)
            else:
                acc = 0.0
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')
        else:
            # Otherwise, just reuse the existing computed accuracy.
            validation_is_true.append(most_likely_answer['accuracy'])

        # Track if the question is answerable.
        validation_answerable.append(is_answerable(example))

        # Keep track of the embedding for the most likely answer to use in p_ik classification later.
        validation_embeddings.append(most_likely_answer['embedding'])

        logging.info('validation_is_true: %f', validation_is_true[-1])

        # If we need predictive entropy or related measures, we compute them here.
        if args.compute_predictive_entropy:
            # log_liks is a list of arrays of token-level log likelihoods for each generation.
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            # Basic sanity check to ensure each generation has token log likelihoods.
            for i in log_liks:
                assert i

            # If we want to test if the context logically entails each generated answer, do that here.
            if args.compute_context_entails_response:
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            # If we condition on the question in a DeBERTa setting, we prepend the question to each response.
            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # We compute "semantic_ids" (clusters) of the responses by using the entailment model to see which
            # responses are logically equivalent or entail each other, effectively cluster them.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            # Store these cluster assignments for each validation sample.
            result_dict['semantic_ids'].append(semantic_ids)

            # Compute an entropy that measures how uncertain the system is about cluster assignment.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Here we compute a "regular entropy" from the average token log-likelihood.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
            entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))

            # We also compute a "semantic entropy" that groups responses by semantic cluster,
            # summing or normalizing the associated log-likelihoods for each cluster, then computing entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            # For debugging/logging, we print out the new counts, cluster IDs, average log-likelihoods, etc.
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        # If computing p_true in this script, we call the relevant function to get the log probability that the answer is true.
        if args.compute_p_true_in_compute_stage:
            #del start
            logging.info("**"*80)
            logging.info("calculate_p_true inputs")
            logging.info("model:")
            logging.info(pt_model)
            logging.info("question:")
            logging.info(question)
            logging.info(" most_likely_answer['response']:")
            logging.info( most_likely_answer['response'])
            logging.info("responses:")
            logging.info(responses)
            logging.info("p_true_few_shot_prompt:")
            logging.info(p_true_few_shot_prompt)
            logging.info("hint old_exp['args'].p_true_hint:")
            logging.info(old_exp['args'].p_true_hint)
            logging.info("**" * 80)
            #del end
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        count += 1
        # If the user only wants a limited number of evaluation samples, break out here.
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    # Compute and log the average accuracy across all validation samples (1.0 means 100% accurate).
    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))

    # Convert the boolean correctness (1.0 vs 0.0) to a "false" measure.
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    # Convert answerability to unanswerability measure, store that as well.
    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    # If we have never populated the 'uncertainty_measures' key, do it now.
    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    # Update the dictionary with the newly computed entropies.
    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    # If we need to compute p_ik or p_ik_answerable, we train a binary classifier on the training embeddings.
    if args.compute_p_ik or args.compute_p_ik_answerable:
        # Prepare training embeddings and labels (correct vs incorrect or answerable vs unanswerable).
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))

        # Convert correctness to a "false" label, and answerability to unanswerable.
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    # If requested, compute p_ik (probability that the answer is incorrect) using embeddings.
    if args.compute_p_ik:
        logging.info('Starting training p_ik on train embeddings.')
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
        logging.info('Finished training p_ik on train embeddings.')

    # If requested, compute p_ik for answerable/unanswerable classification.
    if args.compute_p_ik_answerable:
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    # If we computed p_true, store those results into result_dict as well (p_false = 1 - p_true).
    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        # p_true was stored in its log form, so we store the exponentiated version below as well:
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    # Save updated result_dict (which now includes new uncertainty measures) to a pickle file.
    utils.save(result_dict, 'uncertainty_measures.pkl')

    # Finally, if we used an entailment model, we can save its prediction cache to avoid recomputation in the future.
    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    # Optionally run analyze_run() to get additional logs and aggregate statistics.
    if args.analyze_run:
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    # We parse arguments, restricting to 'compute' stage.
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)