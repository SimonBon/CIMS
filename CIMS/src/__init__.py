celltype_colors = {
    "B":             "#60AD6B",  # green
    "DC":            "#D8B35E",  # muted golden
    "Monocyte":      "#E7C749",  # yellow
    "M1":            "#DD8C39",  # orange
    "M2":            "#A56928",  # brownish orange
    "Neutrophil":    "#77B846",  # light green
    "CD4":           "#26456F",  # dark navy blue
    "CD8":           "#4B76B7",  # mid blue
    "TReg":          "#77A4DD",  # light blue
    "Cytotoxic CD8": "#3A7FEA",  # bright blue
    "NK":            "#5FC3C0",  # teal
    "Lymphatic":     "#C14D48",  # muted red
    "Endothelial":   "#80221B",  # dark red
    "Epithelial":    "#D56D7F",  # salmon pink
    "Tumor":         "#B664B5",  # magenta / purple
    "Mast":          "#8B6EC0",  # lavender
    "Other":         "#606060",  # dark gray
    "Seg Artifact":  "#AFAFAF",  # light gray
}

meta_celltype_colors = {
    "B Cell":         "#60AD6B",  # from B
    "T Cell":         "#26456F",  # from CD4
    "NK Cell":        "#5FC3C0",  # from NK
    "Myeloid":        "#E7C749",  # from Monocyte
    "Dendritic Cell": "#D8B35E",  # from DC
    "Granulocyte":    "#77B846",  # from Neutrophil
    "Mast Cell":      "#8B6EC0",  # from Mast
    "Endothelial":    "#80221B",  # from Endothelial
    "Lymphatic":      "#C14D48",  # from Lymphatic
    "Epithelial":     "#D56D7F",  # from Epithelial
    "Tumor":          "#B664B5",  # from Tumor
    "Other":          "#606060",  # from Other / Seg Artifact
}

meta_map = {
    "CD4": "T Cell",
    "CD8": "T Cell",
    "Cytotoxic CD8": "T Cell",
    "TReg": "T Cell",
    "B": "B Cell",
    "NK": "NK Cell",
    "Monocyte": "Myeloid",
    "M1": "Myeloid",
    "M2": "Myeloid",
    "DC": "Dendritic Cell",
    "Neutrophil": "Granulocyte",
    "Mast": "Mast Cell",
    "Endothelial": "Endothelial",
    "Lymphatic": "Lymphatic",
    "Epithelial": "Epithelial",
    "Tumor": "Tumor",         
    "Other": "Other",
    "Seg Artifact": "Other"
}

