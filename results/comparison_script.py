import json
import matplotlib.pyplot as plt
def main():
    parallel_rust = pull_data_from_json("./parallel.json")
    rust = pull_data_from_json("./rust_results.json")
    python = pull_data_from_json("./py_results.json")
    fig, ax = plt.subplots(figsize=(10, 100))
    b1 = ax.hist(parallel_rust["times"], color="red", label="Parallelised rust")
    b2 = ax.hist(rust["times"], color="orange", label="Non-parallelised rust")
    b3 = ax.hist(python["times"], color="blue", label="Non-parallelised pytorch")
    ax.legend();

    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_title("Copmparison between execution times in different implementations")
    ax.xaxis.set_label_text("Execution time (s)")
    ax.yaxis.set_label_text("Runs")
    plt.show()

def pull_data_from_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    results = data["results"]
    return results[0]

if __name__ == "__main__":
    main()
