from fairseq_cli.hydra_train import cli_main

# from mul.tasks import
# import mul.criterions
from ignisight import tasks,models,criterions
import warnings
warnings.filterwarnings("ignore")

def main():
    cli_main()

if __name__ == "__main__":
    cli_main()
