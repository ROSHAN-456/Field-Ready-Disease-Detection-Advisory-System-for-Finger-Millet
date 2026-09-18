import os
from PIL import Image
from pathlib import Path
import datetime

def generate_quality_report(dataset_dir: str, report_out_path: str):
    """
    Generates a Markdown report summarizing the state of the dataset split.
    Profiles image resolutions and dataset composition.
    """
    base = Path(dataset_dir)
    
    if not base.exists():
        print(f"Directory {dataset_dir} does not exist.")
        return

    splits = ['train', 'val', 'test']
    report_lines = []
    
    report_lines.append(f"# Dataset Quality & Profiling Report")
    report_lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Dataset Location:** `{dataset_dir}`\n")
    report_lines.append(f"## 1. Composition per Split\n")
    
    total_imgs_corpus = 0
    
    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            continue
            
        report_lines.append(f"### {split.capitalize()} Set")
        report_lines.append("| Class | Count | Min Res | Max Res | Avg Res |")
        report_lines.append("|---|---|---|---|---|")
        
        split_total = 0
        classes = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for cls in classes:
            images = [f for f in cls.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
            count = len(images)
            split_total += count
            
            widths = []
            heights = []
            
            for img_path in images:
                try:
                    with Image.open(img_path) as im:
                        widths.append(im.width)
                        heights.append(im.height)
                except:
                    pass
            
            if count > 0:
                min_w, min_h = min(widths), min(heights)
                max_w, max_h = max(widths), max(heights)
                avg_w, avg_h = sum(widths)//count, sum(heights)//count
                res_str = f"| {cls.name} | {count} | {min_w}x{min_h} | {max_w}x{max_h} | {avg_w}x{avg_h} |"
            else:
                res_str = f"| {cls.name} | {count} | N/A | N/A | N/A |"
                
            report_lines.append(res_str)
            
        report_lines.append(f"\n**Total {split} images:** {split_total}\n")
        total_imgs_corpus += split_total

    report_lines.append(f"## 2. Global Statistics")
    report_lines.append(f"- **Total Valid Images:** {total_imgs_corpus}")
    
    with open(report_out_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"Report generated successfully -> {report_out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dataset report")
    parser.add_argument("--input", type=str, default="dataset_split", help="Path to SPLIT dataset (containing train/val/test folders)")
    parser.add_argument("--output", type=str, default="report.md", help="Path for output markdown report (e.g. report.md)")
    args = parser.parse_args()
    
    generate_quality_report(args.input, args.output)
