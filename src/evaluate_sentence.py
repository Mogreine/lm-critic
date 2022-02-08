from argparse import ArgumentParser

from src.critic import LMCritic


def evaluate(sentence: str, is_refined: bool = True, use_gpu: bool = False):
    critic = LMCritic(use_gpu=use_gpu)
    is_good, logp, counter_ex = critic.evaluate_sentence(
        sentence, is_refined=is_refined, verbose=True, return_counter_example=True
    )

    return is_good, logp, counter_ex


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--sent", "-s", type=str, required=True, help="Sentence to evaluate.")
    args.add_argument("--use_gpu", action="store_true")
    args.add_argument("--refined", action="store_true", help="Perturbation method.")
    args = args.parse_args()

    is_good, logp, counter_ex = evaluate(args.sent, args.refined, args.use_gpu)

    print(f"Sentence is good: {is_good}.")
    print(f"Sentence's logp: {logp}.")
    print(f"Counter examples: {counter_ex}")
