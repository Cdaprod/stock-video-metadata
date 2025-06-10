# /scripts/export.py
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

def export_blackbox_csv(df: pd.DataFrame, out_csv: Path):
    """
    Export a CSV with exactly the three columns BlackBox Global requires:
      - Filename
      - Description
      - Keywords
    """
    df[['Filename', 'Description', 'Keywords']].to_csv(out_csv, index=False)
    print(f"✅ Exported CSV: {out_csv}")

def export_blackbox_xml(df: pd.DataFrame, batches_root: Path):
    """
    For each batch_name in the DataFrame, writes a metadata.xml alongside
    that batch's files. Creates the batch directory if it does not exist.
    """
    for batch, group in df.groupby('batch_name'):
        batch_dir = batches_root / batch
        batch_dir.mkdir(parents=True, exist_ok=True)

        root = ET.Element("MediaMetaData")
        for _, r in group.iterrows():
            clip = ET.SubElement(root, "Clip")
            ET.SubElement(clip, "Filename").text    = str(r['Filename'])
            ET.SubElement(clip, "Description").text = str(r['Description'])
            ET.SubElement(clip, "Keywords").text    = str(r['Keywords'])

        xml_path = batch_dir / "metadata.xml"
        tree = ET.ElementTree(root)
        tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)
        print(f"✅ Exported XML for batch `{batch}`: {xml_path}")