"""Deterministic synthetic files for integration tests, not scientific benchmarks."""
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pandas as pd


def create_validation_fixtures(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (1000, 460), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    for x, height, color, label in [(170, 100, '#d62728', 'RED'), (430, 310, '#2468df', 'BLUE'),
                                   (690, 200, '#299447', 'GREEN')]:
        draw.rectangle((x, 370-height, x+140, 370), fill=color)
        draw.text((x+15, 390), label, fill='black', font=font)
    chart_path = folder/'visual-only-chart.png'
    image.save(chart_path)
    logo = Image.new('RGB', (50, 50), '#d62728')
    buffer = BytesIO()
    logo.save(buffer, 'PNG')
    logo_bytes = buffer.getvalue()

    def header(pdf, title, page):
        pdf.setFont('Helvetica-Bold', 17)
        pdf.drawString(55, 742, title)
        pdf.setFont('Helvetica', 9)
        pdf.drawString(55, 35, 'Synthetic integration fixture - not a real research result')
        pdf.drawRightString(555, 35, str(page))

    def paragraph(pdf, text, top):
        import textwrap
        pdf.setFont('Helvetica', 11)
        for i, line in enumerate(textwrap.wrap(text, width=88)):
            pdf.drawString(55, top-i*16, line)

    def table(pdf, title, rows, top):
        pdf.setFont('Helvetica', 11)
        pdf.drawString(55, top, title)
        top -= 18
        width, height = 450, 28
        for r, row in enumerate(rows):
            for col, text in enumerate(row):
                x = 55+col*width/len(row)
                pdf.rect(x, top-(r+1)*height, width/len(row), height)
                pdf.drawString(x+8, top-r*height-18, str(text))

    alpha = folder/'synthetic-aster.pdf'
    pdf = canvas.Canvas(str(alpha), pagesize=(612, 792), invariant=1)
    pdf.setTitle('Aster synthetic research fixture')
    header(pdf, '1 Introduction', 1)
    paragraph(pdf, 'Aster uses a convolutional backbone and a feature pyramid for small object detection. '
              'The controlled test uses the same daylight dataset and evaluation split as Boreal.', 690)
    table(pdf, 'Table 2. Aster controlled detection results.',
          [('Method', 'Small-object AP'), ('Aster', '81.0'), ('Baseline', '72.0')], 570)
    pdf.showPage()
    header(pdf, '2 Visual Results', 2)
    pdf.drawImage(ImageReader(BytesIO(logo_bytes)), 520, 670, width=28, height=28)
    paragraph(pdf, 'The following chart contains the only evidence for the color question. '
              'Bar heights and winning color are deliberately absent from the PDF text.', 690)
    pdf.drawImage(str(chart_path), 55, 405, width=460, height=212)
    pdf.setFont('Helvetica', 11)
    pdf.drawString(55, 380, 'Figure 3. Color-only performance chart with three bars.')
    pdf.setFont('Helvetica-Bold', 17)
    pdf.drawString(55, 280, '3 Discussion')
    paragraph(pdf, 'The diagram illustrates a qualitative ranking. Read the image rather than '
              'inferring the answer from this paragraph or the caption.', 248)
    pdf.showPage()
    header(pdf, '4 Ablation', 3)
    table(pdf, 'Table 4. Component ablation on the controlled dataset.',
          [('Variant', 'Small-object AP'), ('Full Aster', '81.0'), ('No pyramid', '74.0')], 680)
    paragraph(pdf, 'Removing the feature pyramid reduces the measured score under the same test protocol. '
              'Night-time scenes and CPU inference latency were not measured.', 495)
    pdf.save()

    beta = folder/'synthetic-boreal.pdf'
    pdf = canvas.Canvas(str(beta), pagesize=(612, 792), invariant=1)
    header(pdf, '1 Boreal Method', 1)
    paragraph(pdf, 'Boreal uses a transformer backbone with a single prediction head. It reaches '
              '83.0 small-object AP on the same daylight dataset and test split as Aster. '
              'This controlled fixture supports cross-document comparison.', 690)
    table(pdf, 'Table 1. Boreal controlled results.',
          [('Method', 'Small-object AP'), ('Boreal', '83.0'), ('Aster', '81.0')], 570)
    pdf.save()
    frame = pd.DataFrame({'region':['North','South','West'], 'sales':[10,30,20], 'cost':[5,12,8]})
    csv_path, excel_path = folder/'sales.csv', folder/'sales.xlsx'
    frame.to_csv(csv_path, index=False)
    frame.to_excel(excel_path, index=False)
    return {'alpha':alpha, 'beta':beta, 'image':chart_path, 'csv':csv_path, 'xlsx':excel_path}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/pdf')
    args = parser.parse_args()
    print({k:str(v.resolve()) for k,v in create_validation_fixtures(args.output).items()})
