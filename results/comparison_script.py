import json
import matplotlib.pyplot as plt
def main():
    python_data = pull_data_from_json("./py_results.json")
    rust_data = pull_data_from_json("./rust_results.json")
    fig, ax = plt.subplots(figsize=(10, 100))
    vp1 = ax.violinplot(python_data["times"], positions=[1], vert=False,
                         showextrema=False)
    vp2 = ax.violinplot(rust_data["times"], positions=[1], vert=False,
                         showextrema=False)

    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_title("Comparison between execution times of rust version (orange) and python version (blue)")
    ax.xaxis.set_label_text("Execution time (s)")
    plt.show()

def pull_data_from_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    results = data["results"]
    return results[0]



if __name__ == "__main__":
    main()
