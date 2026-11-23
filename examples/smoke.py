"""Минимальная демонстрация. Требуется работающий Ollama с qwen3:14b."""
import argparse
from bica_agent import AbductiveAgent, AgentConfig, OllamaLLM, ScenarioStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["provided", "generate"], default="provided")
    parser.add_argument("--memory-dir", default="./results/agent")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    store = ScenarioStore(args.memory_dir)
    agent = AbductiveAgent(
        OllamaLLM(
            model=args.model,
            host="http://192.168.0.82:11434"
        ),
        store,
        AgentConfig(
            k=args.k,
            n_generate=3,
            evaluate=True,
            self_refine=True
        )
    )

    if args.mode == "provided":
        scenario = store.create_or_load(
            "wet_clothes_choice_clean",
            ["A person returned home wearing wet clothes."],
            ["The person got soaked while walking in the rain.",
             "Water from a garden sprinkler splashed onto the person."],
        )
        facts_to_add = [
            "Throughout that day, there was no atmospheric precipitation at any place the person visited.",
            "The person owns a red notebook.",
        ]
    else:
        scenario = store.create_or_load(
            "wet_clothes_generation_clean",
            ["A person returned home wearing wet clothes."],
        )
        facts_to_add = []

    # Повторный запуск не возвращает последний раунд трижды, а продолжает
    # сценарий с текущей незавершённой ступени.
    for i in range(1 + len(facts_to_add)):
        required = 1 + i
        if len(scenario["observations"]) > required:
            print(f"Раунд {i + 1} уже пройден; пропускаем.")
            continue
        if len(scenario["observations"]) < required:
            store.add_observation(scenario, facts_to_add[i - 1])
        result = agent.run_round(scenario)
        print(f"Раунд {result['round']}: {result['final']}")
        print("Гипотезы:", result["hypotheses"])
        print("Оценки:", result["evaluations"])
        print("Голоса:", result["selection"]["counts"])
        print("Рефлексия:", result["critique"]["verdict"])
        print("Время, с:", result["seconds"], "Токены:", result["tokens"])
        print()


if __name__ == "__main__":
    main()
