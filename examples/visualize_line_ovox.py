"""Make an offline HTML viewer for a saved curve O-Voxel archive."""
import argparse
from pathlib import Path
from o_voxel.convert import LineDualGrid
from o_voxel.convert.line_visualize import write_line_grid_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/line_viewer.html"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_line_grid_html(args.output, LineDualGrid.load(args.input))
    print(f"Open {args.output.resolve()} in a browser")


if __name__ == "__main__":
    main()
