import json
import matplotlib.pyplot as plt
def main():
    parallel = pull_data_from_json("./parallel.json")
    rust_data = pull_data_from_json("./rust_results.json")
    fig, ax = plt.subplots(figsize=(10, 100))
    vp1 = ax.violinplot(parallel["times"], positions=[1], vert=False,
                         showextrema=False)
    for body in vp1['bodies']:
            body.set_facecolor('red')
            body.set_edgecolor('darkred')
            body.set_alpha(0.6) # transparency helps when overlapping
    vp2 = ax.violinplot(rust_data["times"], positions=[1], vert=False,
                         showextrema=False)
    for body in vp2['bodies']:
        body.set_facecolor('orange')
        body.set_edgecolor('darkorange')
        body.set_alpha(0.6)

    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_title("Comparison between execution times of rust version (orange) and rust parallel version (red)")
    ax.xaxis.set_label_text("Execution time (s)")
    plt.show()

def pull_data_from_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    results = data["results"]
    return results[0]



if __name__ == "__main__":
    main()
