import re
from collections import defaultdict

def escape_latex(text):
    """Escapes special LaTeX characters to prevent compilation errors."""
    special_chars = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', 
        '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
    }
    return "".join(special_chars.get(c, c) for c in text)

def generate_latex_table(txt_filepath, tex_filepath):
    # Dictionary to store data: data[layer][head] = [list of texts]
    data = defaultdict(lambda: defaultdict(list))
    
    # Read the text file
    with open(txt_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_layer = None
    current_head = None
    
    # Parse the contents
    for line in lines:
        line = line.strip()
        if not line or line.startswith('---'):
            continue
            
        # Match the Layer and Head headers
        match = re.match(r"Analyzing Layer (\d+) Head (\d+)\.\.\.", line)
        if match:
            current_layer = int(match.group(1))
            current_head = int(match.group(2))
        elif current_layer is not None and current_head is not None:
            # We are on the data line following a header
            items = line.split('||')
            top_texts = []
            
            for item in items:
                if not item.strip():
                    continue
                # Split by ':' and take the first part (the text), then escape it
                raw_text = item.split(':')[0].strip()
                safe_text = escape_latex(raw_text)
                
                top_texts.append(safe_text)
                # Stop after collecting the top 10
                if len(top_texts) == 10:
                    break
            
            data[current_layer][current_head] = top_texts
            
            # Reset trackers to avoid adding stray lines
            current_layer = None
            current_head = None

    # Get sorted layers (e.g., [8, 9, 10, 11]) and heads (0 to 11)
    layers = sorted(data.keys())
    heads = range(12) 
    
    # Generate the LaTeX document
    latex_code = []
    latex_code.append(r"\documentclass{article}")
    latex_code.append(r"\usepackage[margin=0.5in]{geometry}") # Small margins for a large table
    latex_code.append(r"\usepackage{array}")
    latex_code.append(r"\usepackage{makecell}")
    latex_code.append(r"\begin{document}")
    latex_code.append("")
    latex_code.append(r"\begin{table}[htbp]")
    latex_code.append(r"\centering")
    latex_code.append(r"\renewcommand{\arraystretch}{1.5}")
    
    # Create columns dynamically based on number of layers found
    # Using p{width} to allow text wrapping if necessary
    col_width = 16 / len(layers) if len(layers) > 0 else 4
    cols_format = "|" + "|".join([f"p{{{col_width}cm}}"] * len(layers)) + "|"
    
    latex_code.append(f"\\begin{{tabular}}{{{cols_format}}}")
    latex_code.append(r"\hline")
    
    # Header row
    header = " & ".join([f"\\textbf{{Layer {l}}}" for l in layers]) + r" \\"
    latex_code.append(header)
    latex_code.append(r"\hline")
    
    # Data rows (12 rows for 12 heads)
    for h in heads:
        row_cells = []
        for l in layers:
            texts = data[l].get(h, [])
            # Vertically stack the texts using \makecell[t] (top-aligned)
            if texts:
                cell_content = r"\makecell[t]{" + r" \\ ".join(texts) + "}"
            else:
                cell_content = ""
            row_cells.append(cell_content)
            
        row_latex = " & ".join(row_cells) + r" \\"
        latex_code.append(row_latex)
        latex_code.append(r"\hline")
        
    latex_code.append(r"\end{tabular}")
    latex_code.append(r"\caption{Top 10 Concepts per Head by Layer}")
    latex_code.append(r"\label{tab:layer_heads}")
    latex_code.append(r"\end{table}")
    latex_code.append("")
    latex_code.append(r"\end{document}")
    
    # Write to file
    with open(tex_filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_code))
        
    print(f"Successfully generated LaTeX file: {tex_filepath}")

# --- Execution ---
# Replace 'data.txt' with the path to your actual text file
# The script will output 'table.tex' which you can upload directly to Overleaf
txt_pth = "/home/nfm/ViT-Prisma/demos/txt_dicts/original_clip_textspan_dla_based.txt"
latex_pth = txt_pth.replace('.txt', '.tex')
generate_latex_table(txt_pth, latex_pth)