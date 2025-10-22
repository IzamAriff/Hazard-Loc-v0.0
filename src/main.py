# End-to-end pipeline entrypoint

from src.train import train
from src.evaluate import evaluate
from src.reconstruct import run_colmap_reconstruction
from src.visualize import show_point_cloud

def main():
    train()
    evaluate()
    run_colmap_reconstruction()
    show_point_cloud()
if __name__ == '__main__':
    main()
