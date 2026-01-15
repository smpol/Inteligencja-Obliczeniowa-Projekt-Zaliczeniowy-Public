"""Script to download and merge garbage classification datasets from Kaggle."""

import kagglehub
import os
import shutil
from pathlib import Path

def analyze_dataset(path, name):
    print(f"\n🔍 Analyzing {name} at {path}...")
    path = Path(path)
    
    # Try to find the root with classes
    if (path / "garbage_classification").exists():
        classes_root = path / "garbage_classification"
        if (classes_root / "garbage_classification").exists():
             classes_root = classes_root / "garbage_classification"
    elif (path / "Garbage classification").exists():
        classes_root = path / "Garbage classification"
        if (classes_root / "Garbage classification").exists():
             classes_root = classes_root / "Garbage classification"
    else:
        classes_root = path
        
    print(f"   Root found: {classes_root}")
    
    if not classes_root.exists():
        print("   ❌ Root folder does not exist!")
        return None, {}
        
    stats = {}
    total = 0
    for d in classes_root.iterdir():
        if d.is_dir():
            count = len(list(d.glob("*")))
            stats[d.name] = count
            total += count
            
    print(f"   Total images: {total}")
    print(f"   Classes ({len(stats)}): {list(stats.keys())}")
    return classes_root, stats

def main():
    print("Starting dataset download and merge...")
    
    print("Downloading datasets...")
    d1_path = kagglehub.dataset_download("mostafaabla/garbage-classification")
    
    print("\n⬇️  Downloading Dataset 2 (TrashNet - 6 classes)...")
    d2_path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
    
    print("\n⬇️  Downloading Dataset 3 (Hassnain Zaidi)...")
    d3_path = kagglehub.dataset_download("hassnainzaidi/garbage-classification")
    
    # 2. ANALYZE
    root1, stats1 = analyze_dataset(d1_path, "Mostafa Abla")
    root2, stats2 = analyze_dataset(d2_path, "TrashNet (asdasdasasdas)")
    root3, stats3 = analyze_dataset(d3_path, "Hassnain Zaidi")
    
    # 3. MERGE STRATEGY (V4 - PLASTIC+METAL & PAPER+CARDBOARD)
    print("\n🧠 MERGE STRATEGY (V4):")
    print("   Target Classes: glass, textile, metal_plastic, battery, biological, paper, trash")
    
    merged_dir = Path(os.getcwd()) / "merged_dataset"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping Function: All source classes -> Simplified classes
    def get_target_class(cls_name):
        cls = cls_name.lower()
        if 'glass' in cls: return 'glass'
        if cls in ['clothes', 'shoes']: return 'textile'
        if 'battery' in cls or 'batteries' in cls: return 'battery'
        if cls == 'biological': return 'biological'
        if cls in ['cardboard', 'paper']: return 'paper' # MERGED!
        if cls in ['metal', 'plastic']: return 'metal_plastic'
        if cls == 'trash': return 'trash'
        return None # Skip unknown

    # 1. Process Mostafa Abla (Base)
    print(f"\n📂 Processing Mostafa Abla (Base)...")
    for d in root1.iterdir():
        if d.is_dir():
            target = get_target_class(d.name)
            if target:
                dst = merged_dir / target
                dst.mkdir(parents=True, exist_ok=True)
                for f in d.glob("*"):
                    if f.is_file():
                         shutil.copy2(f, dst / f"base_{d.name}_{f.name}")
            else:
                print(f"   ⚠️ Skipping class: {d.name}")

    # 2. Process TrashNet - Mapping 6 -> Simplified
    print("\n🔄 Merging TrashNet...")
    for d in root2.iterdir():
        if d.is_dir():
            target = get_target_class(d.name) 
            if target:
                dst = merged_dir / target
                dst.mkdir(parents=True, exist_ok=True)
                for f in d.glob("*"):
                    if f.is_file():
                        shutil.copy2(f, dst / f"d2_{d.name}_{f.name}")
            else:
                print(f"   ⚠️ Skipping class: {d.name}")

    # 3. Process Hassnain Zaidi (Recursive)
    print("\n🔄 Merging Hassnain Zaidi (Recursive)...")
    subdirs = [x for x in root3.iterdir() if x.is_dir()]
    has_splits = any(x.name in ['train', 'val', 'test'] for x in subdirs)
    
    source_folders = [x for x in subdirs if x.name in ['train', 'val', 'test']] if has_splits else [root3]
        
    for folder in source_folders:
        print(f"   Processing folder: {folder.name}")
        for class_dir in folder.iterdir():
            if not class_dir.is_dir(): continue
            
            target = get_target_class(class_dir.name)
            if target:
                dst = merged_dir / target
                dst.mkdir(parents=True, exist_ok=True)
                
                count = 0
                for f in class_dir.glob("*"):
                    if f.is_file():
                        shutil.copy2(f, dst / f"d3_{folder.name}_{class_dir.name}_{f.name}")
                        count += 1
                # print(f"      + {class_dir.name} -> {target} ({count} images)")
            else:
                 pass # print(f"      SKIP: Unknown class '{class_dir.name}'")
                     
    print("\n✅ FINAL MERGE COMPLETE (SIMPLIFIED)!")
    
    # Count final stats
    final_stats = {}
    for d in merged_dir.iterdir():
        if d.is_dir():
            final_stats[d.name] = len(list(d.glob("*")))
            
    print(f"   Final Classes: {len(final_stats)}")
    for k, v in final_stats.items():
        print(f"   - {k}: {v}")

if __name__ == "__main__":
    main()
