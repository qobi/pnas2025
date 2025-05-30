import os
import sys
import glob
import math
import io
import fitz  # PyMuPDF
from PIL import Image

def pdf_first_page_to_image(pdf_path, zoom=2):
    """
    Extracts the first page of a PDF as a PIL Image.
    
    :param pdf_path: Path to the PDF file.
    :param zoom: Zoom factor for the image resolution.
    :return: PIL Image of the first PDF page.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def create_collage(images, cols):
    """
    Creates a collage from a list of PIL Images.
    
    :param images: List of PIL Images.
    :param cols: Number of columns in the collage.
    :return: PIL Image of the collage.
    """
    if not images:
        print("No images to create a collage.")
        sys.exit(1)
        
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    rows = math.ceil(len(images) / cols)
    collage_width = cols * max_width
    collage_height = rows * max_height
    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255,255,255))
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * max_width
        y = row * max_height
        collage_image.paste(img, (x, y))
    return collage_image

def main():
    pdf_dir = "references/articles/sudb"
    pdf_files = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    if not pdf_files:
        print("No PDF files found in the current directory.")
        sys.exit(1)
    images = []
    for pdf in pdf_files:
        img = pdf_first_page_to_image(pdf)
        if img:
            images.append(img)
    if not images:
        print("No images were extracted from PDFs.")
        sys.exit(1)
    cols = 6  # Number of columns in the collage
    collage_image = create_collage(images, cols)
    collage_image.save('collage.png')
    print("Collage saved as 'collage.png'.")

if __name__ == '__main__':
    main()
