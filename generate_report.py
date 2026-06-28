"""Generate Brain Tumor MRI Detection Project Phase-1 Report (VTU/CIT format)."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

OUTPUT = Path("projectRelated/Brain_Tumor_MRI_Detection_Project_Report_Formatted.docx")

# Margins aligned with the college template (pdf_Cloud_Security_project_cse.docx)
MARGIN_COVER = dict(top=1.0, bottom=1.0, left=0.75, right=0.5)
MARGIN_FRONT = dict(top=0.68, bottom=0.75, left=0.5, right=0.25)
MARGIN_BODY = dict(top=0.65, bottom=0.83, left=0.5, right=0.25)


def set_run_font(run, *, bold=False, italic=False, size=12, name="Times New Roman"):
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def _paragraph_border(paragraph, edge: str, *, sz: str = "6") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    for child in p_pr.findall(qn("w:pBdr")):
        p_pr.remove(child)
    border_wrap = OxmlElement("w:pBdr")
    border = OxmlElement(f"w:{edge}")
    border.set(qn("w:val"), "single")
    border.set(qn("w:sz"), sz)
    border.set(qn("w:space"), "1")
    border.set(qn("w:color"), "000000")
    border_wrap.append(border)
    p_pr.append(border_wrap)


def _add_page_number_field(paragraph) -> None:
    run = paragraph.add_run()
    r = run._r
    for tag, attrs in (
        ("w:fldChar", {"w:fldCharType": "begin"}),
        ("w:instrText", {"xml:space": "preserve"}),
        ("w:fldChar", {"w:fldCharType": "separate"}),
        ("w:fldChar", {"w:fldCharType": "end"}),
    ):
        el = OxmlElement(tag)
        for key, val in attrs.items():
            el.set(qn(key), val)
        if tag == "w:instrText":
            el.text = " PAGE "
        r.append(el)


def configure_section_margins(section, margins: dict) -> None:
    section.top_margin = Inches(margins["top"])
    section.bottom_margin = Inches(margins["bottom"])
    section.left_margin = Inches(margins["left"])
    section.right_margin = Inches(margins["right"])
    section.header_distance = Inches(0.28)
    section.footer_distance = Inches(0.32)


def setup_header(section) -> None:
    section.header.is_linked_to_previous = False
    header = section.header
    hp = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    hp.clear()
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hp.paragraph_format.space_after = Pt(2)
    _paragraph_border(hp, "bottom", sz="8")


def setup_footer(section) -> None:
    section.footer.is_linked_to_previous = False
    footer = section.footer
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    fp.clear()
    fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fp.paragraph_format.space_before = Pt(6)
    fp.paragraph_format.space_after = Pt(0)
    tab_stops = fp.paragraph_format.tab_stops
    tab_stops.add_tab_stop(Inches(3.35), WD_TAB_ALIGNMENT.CENTER)
    tab_stops.add_tab_stop(Inches(6.55), WD_TAB_ALIGNMENT.RIGHT)
    _paragraph_border(fp, "top", sz="12")
    _paragraph_border(fp, "bottom", sz="6")

    left = fp.add_run("B.E., Dept of CSE, CITech")
    set_run_font(left, size=10)
    fp.add_run("\t")
    _add_page_number_field(fp)
    for run in fp.runs[2:]:
        set_run_font(run, size=10)
    right = fp.add_run("\t2025-2026")
    set_run_font(right, size=10)


def apply_section_layout(section, margins: dict, *, header_footer: bool = True) -> None:
    configure_section_margins(section, margins)
    if header_footer:
        setup_header(section)
        setup_footer(section)


def start_new_section(doc, margins: dict, *, header_footer: bool = True):
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    apply_section_layout(section, margins, header_footer=header_footer)
    return section


def add_centered(doc, text, *, bold=False, size=12, style="Normal", space_after=6, space_before=0):
    p = doc.add_paragraph(style=style)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(text)
    set_run_font(run, bold=bold, size=size)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level <= 2 else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(12 if level <= 2 else 8)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(text)
    set_run_font(run, bold=True, size=14 if level == 1 else 13)
    return p


def add_chapter_heading(doc, chapter_no: str, chapter_title: str) -> None:
    p1 = doc.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.space_before = Pt(18)
    p1.paragraph_format.space_after = Pt(4)
    r1 = p1.add_run(chapter_no)
    set_run_font(r1, bold=True, size=14)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_before = Pt(0)
    p2.paragraph_format.space_after = Pt(12)
    r2 = p2.add_run(chapter_title)
    set_run_font(r2, bold=True, size=14)


def add_section_heading(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(text)
    set_run_font(run, bold=True, size=12)


def add_body(doc, text, *, bold=False, justify=True, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if justify else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(text)
    set_run_font(run, bold=bold, size=12)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Paragraph")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Inches(0.45)
    p.paragraph_format.first_line_indent = Inches(-0.2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(f"• {text}")
    set_run_font(run, size=12)
    return p


def add_table(doc, headers, rows, caption=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                set_run_font(r, bold=True, size=11)
    for ri, row in enumerate(rows):
        cells = table.rows[ri + 1].cells
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            for p in cells[ci].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if ci > 0 else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                for r in p.runs:
                    set_run_font(r, size=11)
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(6)
        run = cap.add_run(caption)
        set_run_font(run, bold=True, size=11)
        cap.paragraph_format.space_after = Pt(10)
    doc.add_paragraph()
    return table


def add_page_break(doc):
    doc.add_page_break()


def add_section_break(doc, margins: dict, *, header_footer: bool = True):
    start_new_section(doc, margins, header_footer=header_footer)


def build_title_pages(doc):
    add_centered(doc, "VISVESVARAYA TECHNOLOGICAL UNIVERSITY", size=14, bold=True, style="Title")
    add_centered(doc, '"JNANA SANGAMA", MACHHE, BELAGAVI-590018', size=12)
    doc.add_paragraph()
    add_centered(doc, "Project Phase-1 Report on", size=12, bold=True)
    p = doc.add_paragraph(style="Heading 1")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run("Brain Tumor MRI Detection System")
    set_run_font(r1, bold=True, size=14)
    p.add_run("\n")
    r2 = p.add_run("Using ResNet50, SVM and Grad-CAM")
    set_run_font(r2, bold=True, size=14)
    add_centered(doc, "Submitted in partial fulfillment of the requirements for the VI semester", size=12)
    add_centered(doc, "Bachelor of Engineering", size=13, bold=True)
    add_centered(doc, "in", size=12)
    add_centered(doc, "Computer Science and Engineering", size=13, bold=True)
    add_centered(doc, "of", size=12)
    add_centered(doc, "Visvesvaraya Technological University, Belagavi. by", size=12)
    doc.add_paragraph()
    add_centered(
        doc,
        "[Student Name 1] ([USN 1])    [Student Name 2] ([USN 2])    [Student Name 3] ([USN 3])",
        size=12,
        bold=True,
    )
    doc.add_paragraph()
    for line in [
        "Under the Guidance of",
        "[Guide Name]",
        "Assistant Professor",
        "Dept. of CSE",
    ]:
        add_centered(doc, line, size=12, bold=True)
    doc.add_paragraph()
    add_centered(doc, "Department of Computer Science and Engineering", size=12, bold=True)
    add_centered(doc, "CAMBRIDGE INSTITUTE OF TECHNOLOGY", size=12, bold=True)
    add_centered(doc, "AN AUTONOMOUS INSTITUTION AFFILIATED TO VTU,", size=12, bold=True)
    add_centered(doc, "BANGALORE - 560 036", size=13, bold=True)
    add_centered(doc, "2025-2026", size=12, bold=True)

    add_section_break(doc, MARGIN_FRONT, header_footer=True)

    add_centered(doc, "CAMBRIDGE INSTITUTE OF TECHNOLOGY", size=14, bold=True, style="Title")
    add_centered(doc, "AN AUTONOMOUS INSTITUTION AFFILIATED TO VTU", size=13, bold=True)
    add_centered(doc, "K.R.Puram, Bangalore-560 036", size=12)
    add_centered(doc, "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING", size=12, bold=True)
    doc.add_paragraph()

    add_heading(doc, "CERTIFICATE", level=1)
    doc.add_paragraph()
    add_body(
        doc,
        "Certified that [Student Name 1], [Student Name 2] and [Student Name 3] bearing "
        "USN: [USN 1], [USN 2] and [USN 3] respectively, students of VI semester BE in "
        "Computer Science and Engineering, have carried out the project work titled "
        '"Brain Tumor MRI Detection System Using ResNet50, SVM and Grad-CAM" '
        "under my guidance during the academic year 2025-2026. The project report has "
        "been submitted to Visvesvaraya Technological University, Belagavi in partial "
        "fulfillment of the requirements for the award of Bachelor of Engineering degree.",
    )
    for _ in range(4):
        doc.add_paragraph()
    add_centered(doc, "-----------------------------", size=12)
    add_centered(doc, "Project Guide", size=12)
    add_centered(doc, "Prof./Dr./Ms./Mr. [Guide Name]", size=12)
    add_centered(doc, "Dept. of CSE, CITech", size=12)
    doc.add_paragraph()
    add_centered(doc, "-----------------------------", size=12)
    add_centered(doc, "Project Co-ordinator", size=12)
    add_centered(doc, "Ms. Lakshmi Shree M S", size=12)
    add_centered(doc, "Dept. of CSE, CITech", size=12)
    doc.add_paragraph()
    add_centered(doc, "--------------------------", size=12)
    add_centered(doc, "Head of the Dept.", size=12)
    add_centered(doc, "Dr. Shreekanth M Prabhu", size=12)
    add_centered(doc, "Dept. of CSE, CITech", size=12)

    add_page_break(doc)

    add_centered(doc, "DECLARATION", size=14, bold=True)
    doc.add_paragraph()
    add_body(
        doc,
        "We, [Student Name 1], [Student Name 2] and [Student Name 3] bearing USN "
        "[USN 1], [USN 2] and [USN 3] respectively, hereby declare that the project "
        "work titled \"Brain Tumor MRI Detection System Using ResNet50, SVM and Grad-CAM\" "
        "submitted to Visvesvaraya Technological University, Belagavi is a record of an "
        "original work done by us under the guidance of [Guide Name], Assistant Professor, "
        "Department of Computer Science and Engineering, Cambridge Institute of Technology, "
        "Bangalore.",
    )
    add_body(
        doc,
        "We also declare that, to the best of our knowledge and belief, the work reported "
        "here does not form part of any other report on the basis of which a degree or "
        "award was conferred on an earlier occasion on us or any other candidate.",
    )
    doc.add_paragraph()
    add_body(doc, "Date:\t\t\tPlace: Bangalore", justify=False)
    doc.add_paragraph()
    add_body(
        doc,
        "[Student Name 1] ([USN 1])\n[Student Name 2] ([USN 2])\n[Student Name 3] ([USN 3])",
        justify=False,
    )

    add_page_break(doc)

    add_heading(doc, "ACKNOWLEDGEMENT", level=1)
    add_body(
        doc,
        "We would like to place on record our deep sense of gratitude to Shri. D. K. Mohan, "
        "Chairman, Cambridge Group of Institutions, Bangalore, India, for providing us the "
        "necessary facilities and an excellent academic environment to carry out this project.",
    )
    add_body(
        doc,
        "We are extremely thankful to Prof. B V Ravishankar, Principal, CITech, Bangalore, "
        "for providing us the academic ambience and everlasting moral support throughout our "
        "project work.",
    )
    add_body(
        doc,
        "We express our sincere gratitude to Dr. Shreekanth Prabhu M, HOD, Dept. of Computer "
        "Science and Engineering, CITech, Bangalore, for his stimulating guidance and constant "
        "encouragement.",
    )
    add_body(
        doc,
        "We wish to extend our thanks to Prof. Lakshmi Shree M S, Project Coordinator, "
        "Dept. of CSE, CITech, Bangalore, for her critical, insightful comments and valuable "
        "suggestions during the course of this project.",
    )
    add_body(
        doc,
        "We also wish to extend our thanks to [Guide Name], Assistant Professor, Dept. of CSE, "
        "CITech for his/her guidance and impressive technical support throughout the development "
        "of this Brain Tumor MRI Detection System.",
    )
    add_body(
        doc,
        "Finally, to all our friends and classmates who always stood by us in difficult "
        "situations and helped us in technical aspects, and last but not least, our parents "
        "for their blessings and moral support.",
    )
    doc.add_paragraph()
    add_body(
        doc,
        "[Student Name 1] ([USN 1])\n[Student Name 2] ([USN 2])\n[Student Name 3] ([USN 3])",
        justify=False,
    )


def build_abstract_and_toc(doc):
    add_page_break(doc)
    add_heading(doc, "ABSTRACT", level=1)
    add_body(
        doc,
        "Brain tumors are among the most critical neurological conditions, and early detection "
        "through Magnetic Resonance Imaging (MRI) plays a vital role in improving patient "
        "outcomes. Manual analysis of MRI scans is time-consuming, subjective, and dependent "
        "on radiologist expertise. This project addresses these challenges by developing an "
        "intelligent Brain Tumor MRI Detection System that automatically classifies MRI scans "
        "into four categories: glioma, meningioma, pituitary tumor, and no tumor.",
    )
    add_body(
        doc,
        "The system employs a hybrid deep learning and machine learning architecture. A "
        "pre-trained ResNet50 Convolutional Neural Network (CNN) is used as a feature "
        "extractor to convert each MRI image into a 2048-dimensional feature vector. A "
        "Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel then classifies "
        "these features. For explainability, Gradient-weighted Class Activation Mapping "
        "(Grad-CAM) is implemented on a parallel CNN model to generate visual heatmaps "
        "highlighting the brain regions that influenced the prediction.",
    )
    add_body(
        doc,
        "The Kaggle Brain Tumor MRI dataset containing 7,200 images (5,600 training and "
        "1,600 testing) is used for training and evaluation. The dataset is balanced across "
        "all four classes. The trained models are deployed through a Flask web application "
        "that allows users to upload MRI images and receive instant predictions with "
        "confidence scores and Grad-CAM visual explanations.",
    )
    add_body(
        doc,
        "The proposed system demonstrates how transfer learning, classical machine learning, "
        "and Explainable AI (XAI) can be combined to build a practical medical image "
        "classification solution suitable for academic research and demonstration purposes.",
    )

    add_page_break(doc)
    add_heading(doc, "CONTENTS", level=1)
    toc_rows = [
        ("Abstract", "i"),
        ("Contents", "ii"),
        ("List of Figures", "iii"),
        ("List of Tables", "iv"),
        ("Chapter 1 — Introduction", "1"),
        ("    Background", "1"),
        ("    Objectives", "2"),
        ("    Scope of the Project", "3"),
        ("    Significance of the Study", "4"),
        ("Chapter 2 — Literature Survey", "5"),
        ("Chapter 3 — System Analysis", "9"),
        ("    Proposed System", "9"),
        ("    System Workflow", "10"),
        ("    Comparison with Existing Systems", "11"),
        ("    Software Requirements", "12"),
        ("    Hardware Requirements", "13"),
        ("    User Requirements", "13"),
        ("    Product Requirements", "14"),
        ("Chapter 4 — Design", "15"),
        ("    Purpose", "15"),
        ("    Architecture Diagram", "16"),
        ("    Data Flow Diagram", "17"),
        ("    Project File Structure", "18"),
        ("Chapter 5 — Expected Outcomes", "19"),
        ("References", "21"),
        ("SDG Map", "22"),
    ]
    toc_header = doc.add_paragraph()
    toc_header.paragraph_format.tab_stops.add_tab_stop(Inches(6.2), WD_TAB_ALIGNMENT.RIGHT)
    hr = toc_header.add_run("Chapters\tPage No.")
    set_run_font(hr, bold=True, size=12)

    for title, page in toc_rows:
        p = doc.add_paragraph()
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.2), WD_TAB_ALIGNMENT.RIGHT)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        run = p.add_run(f"{title}\t{page}")
        set_run_font(run, size=12)

    add_page_break(doc)
    add_heading(doc, "LIST OF FIGURES", level=1)
    add_table(
        doc,
        ["Figure No.", "Figure Name", "Page No."],
        [
            ("4.1", "Architecture Diagram of Brain Tumor MRI Detection System", "16"),
            ("4.2", "Data Flow Diagram of MRI Classification Pipeline", "17"),
        ],
    )

    add_heading(doc, "LIST OF TABLES", level=1)
    add_table(
        doc,
        ["Table No.", "Table Name", "Page No."],
        [
            ("2.1", "Comparative Analysis of Reviewed Literature", "8"),
            ("3.1", "Software Requirements", "12"),
            ("3.2", "Hardware Requirements", "13"),
            ("3.3", "Classification Classes", "10"),
            ("4.1", "Project File Structure", "18"),
            ("6.1", "Sustainable Development Goals (SDG) Alignment Matrix", "22"),
        ],
    )


def build_chapter1(doc):
    add_section_break(doc, MARGIN_BODY, header_footer=True)
    add_chapter_heading(doc, "CHAPTER 1", "INTRODUCTION")

    add_section_heading(doc, "1.1 Background")
    add_body(
        doc,
        "Brain tumors represent one of the most serious health challenges worldwide. According "
        "to medical research, early and accurate detection significantly improves treatment "
        "planning and patient survival rates. Magnetic Resonance Imaging (MRI) is the primary "
        "non-invasive diagnostic tool used by neurologists and radiologists to identify and "
        "classify brain abnormalities.",
    )
    add_body(
        doc,
        "Traditional diagnosis relies heavily on manual examination of MRI scans by trained "
        "radiologists. This process is labor-intensive, time-consuming, and can be subject to "
        "inter-observer variability. As the volume of medical imaging data continues to grow, "
        "there is an increasing need for automated computer-aided diagnosis (CAD) systems "
        "that can assist medical professionals in rapid and consistent tumor classification.",
    )
    add_body(
        doc,
        "Deep Learning, particularly Convolutional Neural Networks (CNNs), has revolutionized "
        "medical image analysis by automatically learning hierarchical visual features from "
        "images. However, training a full CNN classifier from scratch requires large annotated "
        "datasets and significant computational resources. An alternative hybrid approach "
        "combines CNN-based feature extraction with classical machine learning classifiers "
        "such as Support Vector Machines (SVM), offering strong performance with reduced "
        "training complexity.",
    )
    add_body(
        doc,
        "This project develops a Brain Tumor MRI Detection System that classifies MRI scans "
        "into glioma, meningioma, pituitary tumor, or no tumor using ResNet50 for feature "
        "extraction, SVM for classification, Grad-CAM for visual explainability, and Flask "
        "for web-based deployment.",
    )

    add_section_heading(doc, "1.2 Objectives")
    add_body(doc, "The primary objectives of this project are:")
    objectives = [
        "To design and implement an automated brain tumor classification system using MRI images.",
        "To extract deep visual features from MRI scans using a pre-trained ResNet50 CNN with ImageNet transfer learning.",
        "To train an SVM classifier with RBF kernel on extracted features for multi-class tumor classification.",
        "To implement Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explainability of model predictions.",
        "To develop a Flask-based web application for real-time MRI upload, prediction, and result visualization.",
        "To evaluate the system on the Kaggle Brain Tumor MRI dataset with balanced training and testing splits.",
        "To demonstrate the integration of deep learning, classical ML, and Explainable AI in a single medical imaging pipeline.",
    ]
    for obj in objectives:
        add_bullet(doc, obj)

    add_section_heading(doc, "1.3 Scope of the Project")
    add_body(
        doc,
        "The scope of this project encompasses the design, development, training, and deployment "
        "of a Brain Tumor MRI Detection System. The system classifies 2D MRI images into four "
        "categories: glioma, meningioma, pituitary tumor, and no tumor.",
    )
    add_body(
        doc,
        "The project uses the Kaggle Brain Tumor MRI dataset containing 7,200 images organized "
        "in folder-based class labels — 5,600 images for training (1,400 per class) and 1,600 "
        "images for testing (400 per class). The dataset is perfectly balanced across all "
        "four classes.",
    )
    add_body(
        doc,
        "The machine learning pipeline consists of three stages: (1) ResNet50 feature extraction "
        "producing 2048-dimensional vectors, (2) SVM training with StandardScaler normalization, "
        "and (3) CNN head training solely for Grad-CAM explainability. Data augmentation "
        "(horizontal flip, rotation, zoom, contrast) is applied during CNN training.",
    )
    add_body(
        doc,
        "The deployment component is a Flask web application with a modern responsive UI that "
        "accepts MRI image uploads, displays predicted class, confidence percentage, per-class "
        "probability bars, original image, and Grad-CAM heatmap overlay.",
    )
    add_body(
        doc,
        "The project is intended as an academic research and demonstration prototype. It is not "
        "designed for clinical deployment without further validation, regulatory approval, and "
        "integration with hospital Picture Archiving and Communication Systems (PACS).",
    )

    add_section_heading(doc, "1.4 Significance of the Study")
    add_body(
        doc,
        "The significance of this study lies in its contribution to the growing field of "
        "AI-assisted medical image analysis and Explainable AI in healthcare.",
    )
    add_body(
        doc,
        "First, the hybrid CNN + SVM approach demonstrates that transfer learning with "
        "pre-trained ResNet50 features can achieve effective tumor classification without "
        "the computational cost of fine-tuning an entire deep network. This makes the approach "
        "accessible for academic institutions with limited GPU resources.",
    )
    add_body(
        doc,
        "Second, the integration of Grad-CAM addresses a critical requirement in medical AI — "
        "explainability. Doctors and reviewers need to understand not just what the model "
        "predicted, but which regions of the MRI influenced that prediction. Grad-CAM heatmaps "
        "provide this visual transparency.",
    )
    add_body(
        doc,
        "Third, the Flask web deployment makes the system practically demonstrable during "
        "project reviews, vivas, and presentations. Users can upload real MRI images and "
        "receive instant results without command-line interaction.",
    )
    add_body(
        doc,
        "Fourth, the project demonstrates how open-source technologies (TensorFlow, scikit-learn, "
        "Flask, OpenCV) can be combined to produce a complete end-to-end medical AI pipeline "
        "from dataset loading through model training to web deployment.",
    )


def build_chapter2(doc):
    add_page_break(doc)
    add_chapter_heading(doc, "CHAPTER 2", "LITERATURE SURVEY")

    add_body(
        doc,
        "Kermany et al. (2018) — Identifying Medical Diagnoses and Treatable Diseases by "
        "Image-Based Deep Learning",
        bold=True,
    )
    add_body(
        doc,
        "This landmark study demonstrated that deep learning models trained on retinal images "
        "could achieve diagnostic performance comparable to expert physicians. The authors "
        "used transfer learning with pre-trained CNNs and showed that medical image "
        "classification benefits significantly from features learned on large-scale datasets. "
        "Key contribution: established the viability of CNN-based medical diagnosis.",
    )
    add_body(
        doc,
        "Key limitations: focused on retinal images, not brain MRI. Required large curated "
        "datasets and end-to-end CNN training. Our project extends this transfer learning "
        "concept to brain MRI using ResNet50 with a hybrid SVM classifier.",
    )

    add_body(
        doc,
        "Deepak and Ameer (2019) — Brain Tumor Classification Using Deep CNN Features "
        "and Machine Learning",
        bold=True,
    )
    add_body(
        doc,
        "This paper proposed a hybrid approach for brain tumor classification where CNN "
        "features are extracted and fed to classical ML classifiers. The authors demonstrated "
        "that combining deep feature extraction with SVM or Random Forest can achieve "
        "competitive accuracy on brain MRI datasets with reduced training time compared to "
        "full end-to-end deep learning.",
    )
    add_body(
        doc,
        "Key limitations: limited explainability — no visual heatmap generation. Our project "
        "addresses this by integrating Grad-CAM for spatial explanation of predictions.",
    )

    add_body(
        doc,
        "Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks via "
        "Gradient-based Localization",
        bold=True,
    )
    add_body(
        doc,
        "This paper introduced Grad-CAM, a technique for producing visual explanations of "
        "CNN decisions without modifying the architecture. By computing gradients of the "
        "target class score with respect to convolutional feature maps, Grad-CAM generates "
        "heatmaps highlighting important image regions. It has become a standard tool in "
        "Explainable AI for medical imaging.",
    )
    add_body(
        doc,
        "Key contribution: enables interpretability in deep learning models. Our project "
        "directly implements Grad-CAM to show which brain regions the model focuses on "
        "during tumor classification.",
    )

    add_body(
        doc,
        "He et al. (2016) — Deep Residual Learning for Image Recognition (ResNet)",
        bold=True,
    )
    add_body(
        doc,
        "ResNet introduced residual connections (skip connections) that enable training of "
        "very deep networks (50, 101, 152 layers) without vanishing gradient problems. "
        "ResNet50 pre-trained on ImageNet has become the standard backbone for transfer "
        "learning in computer vision and medical image analysis tasks.",
    )
    add_body(
        doc,
        "Our project uses ResNet50 as a frozen feature extractor, leveraging ImageNet "
        "pre-trained weights to extract 2048-dimensional feature vectors from MRI images.",
    )

    add_section_heading(doc, "2.1 Comparative Analysis of Reviewed Literature")
    add_body(
        doc,
        "The reviewed literature collectively establishes the foundation for our hybrid "
        "architecture. Table 2.1 summarises the key contributions and gaps addressed.",
    )
    add_table(
        doc,
        ["Paper", "Approach", "Limitation", "Our Contribution"],
        [
            (
                "Kermany et al. (2018)",
                "End-to-end CNN",
                "Requires large data & GPU",
                "Hybrid CNN+SVM reduces training cost",
            ),
            (
                "Deepak & Ameer (2019)",
                "CNN features + ML",
                "No explainability",
                "Grad-CAM heatmap integration",
            ),
            (
                "Selvaraju et al. (2017)",
                "Grad-CAM XAI",
                "Needs spatial CNN maps",
                "Dedicated CNN head for Grad-CAM",
            ),
            (
                "He et al. (2016)",
                "ResNet50 backbone",
                "General vision, not medical",
                "Transfer learning on MRI data",
            ),
        ],
        caption="Table 2.1: Comparative Analysis of Reviewed Literature",
    )


def build_chapter3(doc):
    add_page_break(doc)
    add_chapter_heading(doc, "CHAPTER 3", "SYSTEM ANALYSIS")

    add_section_heading(doc, "3.1 Proposed System")
    add_body(
        doc,
        "The proposed system is a Brain Tumor MRI Detection System that accepts brain MRI "
        "images as input and classifies them into one of four categories: glioma, meningioma, "
        "pituitary tumor, or no tumor. The system uses a hybrid deep learning and machine "
        "learning pipeline combined with visual explainability and web-based deployment.",
    )
    add_body(
        doc,
        "During training, MRI images are loaded from folder-structured datasets, preprocessed "
        "to 224×224 resolution, and passed through a frozen ResNet50 network to extract "
        "2048-dimensional feature vectors. These features are normalized using StandardScaler "
        "and used to train an SVM classifier with RBF kernel. A separate CNN model with a "
        "softmax classification head is trained for Grad-CAM explainability.",
    )
    add_body(
        doc,
        "During inference, a user uploads an MRI image through the Flask web interface. The "
        "image is preprocessed, features are extracted via ResNet50, the SVM predicts the "
        "tumor class with confidence scores, and Grad-CAM generates a heatmap overlay "
        "showing the influential brain regions.",
    )

    add_section_heading(doc, "3.2 System Workflow")
    add_body(doc, "The system operates through the following workflow:")
    workflow = [
        "User opens the Flask web application at http://127.0.0.1:5000/",
        "User uploads a brain MRI image (JPG/PNG) via the web form or drag-and-drop interface.",
        "Server loads the image, converts to RGB, and resizes to 224×224 pixels.",
        "Image is preprocessed using ResNet-specific preprocess_input normalization.",
        "ResNet50 feature extractor converts the image to a 2048-dimensional feature vector.",
        "SVM Pipeline (StandardScaler + SVC) predicts class probabilities for all four categories.",
        "Argmax selects the predicted class; confidence score is displayed to the user.",
        "Grad-CAM generates a heatmap on the parallel CNN model for the predicted class.",
        "Heatmap is overlaid on the original MRI using JET colormap and alpha blending.",
        "Results page displays prediction, confidence, probability bars, original image, and Grad-CAM overlay.",
    ]
    for i, step in enumerate(workflow, 1):
        add_bullet(doc, f"Step {i}: {step}")

    add_section_heading(doc, "3.3 Classification Classes")
    add_table(
        doc,
        ["Class Label", "Description", "Training Images", "Testing Images"],
        [
            ("glioma", "Malignant tumor from glial cells", "1,400", "400"),
            ("meningioma", "Tumor in brain/spinal cord lining", "1,400", "400"),
            ("pituitary", "Tumor in pituitary gland", "1,400", "400"),
            ("notumor", "Healthy brain — no tumor", "1,400", "400"),
            ("Total", "—", "5,600", "1,600"),
        ],
        caption="Table 3.3: Classification Classes and Dataset Distribution",
    )

    add_section_heading(doc, "3.4 Comparison with Existing Systems")
    add_body(
        doc,
        "Traditional brain tumor diagnosis relies entirely on manual radiologist examination, "
        "which is slow and subjective. Pure deep learning approaches require extensive GPU "
        "resources and large annotated datasets for end-to-end training. Our hybrid approach "
        "offers a practical middle ground.",
    )
    add_body(doc, "Primary Features of the Proposed System:")
    features = [
        "Hybrid ResNet50 + SVM architecture for efficient and accurate classification.",
        "Transfer learning from ImageNet — no need to train CNN from scratch.",
        "Grad-CAM visual explainability showing brain regions influencing predictions.",
        "Flask web application with drag-and-drop upload and real-time inference.",
        "Per-class probability display with confidence percentage and ranked results.",
        "Modular codebase: separate files for training, inference, utilities, and Grad-CAM.",
        "Saved model artifacts (svm.joblib, cnn_classifier.keras) for instant deployment.",
        "Balanced dataset handling with class_weight='balanced' in SVM.",
    ]
    for f in features:
        add_bullet(doc, f)

    add_section_heading(doc, "3.5 Software Requirements")
    add_table(
        doc,
        ["Software", "Version", "Purpose"],
        [
            ("Python", "3.10 or higher", "Primary programming language"),
            ("TensorFlow / Keras", "≥ 2.15", "Deep learning — ResNet50, training, Grad-CAM"),
            ("scikit-learn", "≥ 1.4", "SVM classifier, StandardScaler, Pipeline"),
            ("NumPy", "≥ 1.26", "Numerical arrays for features and images"),
            ("Pillow (PIL)", "≥ 10.2", "Image loading, resizing, PNG encoding"),
            ("OpenCV", "≥ 4.9", "Heatmap resize, JET colormap, overlay blending"),
            ("Flask", "≥ 3.0", "Web framework for deployment"),
            ("joblib", "≥ 1.3", "SVM model serialization and loading"),
            ("Matplotlib", "≥ 3.8", "Plotting support (optional visualization)"),
            ("VS Code / Cursor", "Latest", "Development IDE"),
            ("HTML/CSS/JavaScript", "—", "Frontend web interface"),
        ],
        caption="Table 3.1: Software Requirements",
    )

    add_section_heading(doc, "3.6 Hardware Requirements")
    add_table(
        doc,
        ["Component", "Minimum Requirement"],
        [
            ("Operating System", "Windows 10/11 or Linux Ubuntu"),
            ("Processor", "Intel i5 or higher (multi-core recommended)"),
            ("RAM", "8 GB minimum (16 GB recommended for TensorFlow)"),
            ("Storage", "10 GB free space for dataset, models, and dependencies"),
            ("GPU", "Optional — CPU training supported; GPU accelerates CNN training"),
            ("Display", "1280×720 minimum for web interface"),
        ],
        caption="Table 3.2: Hardware Requirements",
    )

    add_section_heading(doc, "3.7 User Requirements")
    user_reqs = [
        "Basic understanding of Python and running scripts in terminal.",
        "Ability to navigate a web browser to access the Flask application.",
        "Understanding of MRI brain scan images for result interpretation.",
        "Familiarity with VS Code or Cursor IDE for running and monitoring the application.",
        "Knowledge of virtual environment setup (python -m venv) for dependency management.",
    ]
    for r in user_reqs:
        add_bullet(doc, r)

    add_section_heading(doc, "3.8 Product Requirements")
    product_reqs = [
        "Pre-trained model artifacts: svm.joblib, class_names.json, cnn_classifier.keras.",
        "Modular codebase with separate files: train.py, app.py, ml_utils.py, gradcam.py.",
        "Kaggle Brain Tumor MRI dataset in dataset/Training and dataset/Testing folders.",
        "Flask web application with file upload and result rendering.",
        "Grad-CAM heatmap generation with JET colormap overlay.",
        "requirements.txt for reproducible dependency installation.",
    ]
    for r in product_reqs:
        add_bullet(doc, r)


def build_chapter4(doc):
    add_page_break(doc)
    add_chapter_heading(doc, "CHAPTER 4", "DESIGN")

    add_section_heading(doc, "4.1 Purpose")
    add_body(
        doc,
        "The purpose of this project is to develop an intelligent Brain Tumor MRI Detection "
        "System that automates the classification of brain MRI scans into tumor categories, "
        "provides confidence scores and visual explanations, and deploys the solution as an "
        "accessible web application.",
    )
    add_body(
        doc,
        "The system employs a hybrid machine learning framework combining a frozen ResNet50 "
        "CNN for feature extraction and an SVM classifier for final prediction. A parallel "
        "CNN model with a softmax head enables Grad-CAM explainability. This dual-model "
        "design separates classification accuracy from visual interpretability.",
    )
    add_body(
        doc,
        "The Flask web interface reduces the technical barrier for users, enabling MRI upload, "
        "instant prediction, and visual result display without command-line interaction.",
    )

    add_section_heading(doc, "4.2 Architecture Diagram")
    arch_layers = [
        ("LAYER 1: DATA LAYER", [
            "Kaggle Brain Tumor MRI Dataset",
            "dataset/Training/ (5,600 images) | dataset/Testing/ (1,600 images)",
            "4 Classes: glioma | meningioma | pituitary | notumor",
        ]),
        ("LAYER 2: FEATURE EXTRACTION LAYER", [
            "train.py → build_feature_extractor()",
            "ResNet50 (ImageNet weights, frozen, pooling=avg)",
            "Output: 2048-dimensional feature vector per image",
        ]),
        ("LAYER 3: CLASSIFICATION LAYER", [
            "StandardScaler → SVM (RBF kernel, probability=True)",
            "Pipeline saved as artifacts/svm.joblib",
            "Output: predicted class + probability scores",
        ]),
        ("LAYER 4: EXPLAINABILITY LAYER", [
            "gradcam.py → make_gradcam_heatmap()",
            "CNN Model: ResNet50 + GAP + Dropout + Dense(4, softmax)",
            "Grad-CAM heatmap → JET colormap → alpha overlay on MRI",
        ]),
        ("LAYER 5: PRESENTATION LAYER", [
            "app.py (Flask Web Server)",
            "templates/index.html",
            "Upload MRI | View Prediction | Confidence Bars | Grad-CAM Overlay",
        ]),
    ]
    for layer_title, items in arch_layers:
        p = doc.add_paragraph()
        run = p.add_run(layer_title)
        set_run_font(run, bold=True, size=12)
        for item in items:
            add_bullet(doc, item)
    add_centered(doc, "Fig 4.1: Architecture Diagram of Brain Tumor MRI Detection System", bold=True, size=11)
    add_body(
        doc,
        "Figure 4.1 illustrates the overall five-layer architecture of the Brain Tumor MRI "
        "Detection System, from dataset ingestion through feature extraction, classification, "
        "explainability, and web presentation.",
    )

    add_section_heading(doc, "4.3 Data Flow Diagram")
    add_body(doc, "The data flow through the system follows a sequential pipeline:")
    dfd = [
        "TRAINING FLOW: Dataset → Preprocess (224×224) → ResNet50 Features (2048-d) → StandardScaler → SVM Train → svm.joblib",
        "TRAINING FLOW (parallel): Dataset → Augment → ResNet50 + CNN Head → Train (Adam, 5 epochs) → cnn_classifier.keras",
        "INFERENCE FLOW: User Upload → PIL Load/Resize → preprocess_input → ResNet50 → SVM predict_proba → Display Results",
        "EXPLAINABILITY FLOW: Preprocessed Image → CNN Model → Grad-CAM Gradients → Heatmap → OpenCV Overlay → Base64 PNG → HTML",
    ]
    for step in dfd:
        add_bullet(doc, step)
    add_centered(doc, "Fig 4.2: Data Flow Diagram of MRI Classification Pipeline", bold=True, size=11)

    add_section_heading(doc, "4.4 Project File Structure")
    add_table(
        doc,
        ["File / Folder", "Purpose"],
        [
            ("train.py", "Full 3-stage training pipeline; saves all artifacts"),
            ("app.py", "Flask web server with inference and Grad-CAM routes"),
            ("ml_utils.py", "Shared utilities: JSON I/O, directory creation, image loading"),
            ("gradcam.py", "Grad-CAM heatmap generation and overlay visualization"),
            ("templates/index.html", "Web UI: upload form, results, probability bars, images"),
            ("artifacts/svm.joblib", "Saved SVM Pipeline (StandardScaler + SVC)"),
            ("artifacts/class_names.json", "Class label list: glioma, meningioma, notumor, pituitary"),
            ("artifacts/cnn_classifier.keras", "CNN model for Grad-CAM explainability"),
            ("dataset/Training/", "5,600 training MRI images in class subfolders"),
            ("dataset/Testing/", "1,600 testing MRI images in class subfolders"),
            ("requirements.txt", "Python package dependencies"),
        ],
        caption="Table 4.1: Project File Structure",
    )

    add_section_heading(doc, "4.5 Key Algorithms Used")
    add_table(
        doc,
        ["Algorithm", "Module", "Purpose"],
        [
            ("ResNet50 CNN", "TensorFlow/Keras", "Deep feature extraction (2048-d vectors)"),
            ("Transfer Learning", "TensorFlow/Keras", "ImageNet pre-trained weights, frozen"),
            ("StandardScaler", "scikit-learn", "Z-score feature normalization"),
            ("SVM (RBF Kernel)", "scikit-learn", "Multi-class tumor classification"),
            ("Platt Scaling", "scikit-learn", "SVM probability calibration"),
            ("Adam Optimizer", "TensorFlow/Keras", "CNN head training"),
            ("Cross-Entropy Loss", "TensorFlow/Keras", "CNN classification loss"),
            ("Grad-CAM", "Custom (gradcam.py)", "Visual explainability heatmaps"),
            ("Data Augmentation", "TensorFlow/Keras", "Flip, rotation, zoom, contrast"),
            ("Argmax", "NumPy", "Final class selection from probabilities"),
        ],
    )


def build_chapter5(doc):
    add_page_break(doc)
    add_chapter_heading(doc, "CHAPTER 5", "EXPECTED OUTCOMES")

    outcomes = [
        (
            "5.1 Functional MRI Classification System",
            "The primary expected outcome is a fully functional Brain Tumor MRI Detection "
            "System that classifies uploaded MRI images into glioma, meningioma, pituitary "
            "tumor, or no tumor with associated confidence scores. The system operates "
            "through a user-friendly Flask web interface requiring no command-line expertise.",
        ),
        (
            "5.2 Hybrid CNN + SVM Classification",
            "The system demonstrates the effectiveness of combining deep learning feature "
            "extraction with classical machine learning classification. ResNet50 extracts "
            "rich visual features while SVM provides efficient, interpretable multi-class "
            "decision boundaries in the 2048-dimensional feature space.",
        ),
        (
            "5.3 Visual Explainability via Grad-CAM",
            "Grad-CAM heatmaps are expected to highlight brain regions that most influenced "
            "the model's prediction. This addresses the critical need for explainability in "
            "medical AI, allowing reviewers and users to visually verify model focus areas.",
        ),
        (
            "5.4 Web-Based Deployment",
            "The Flask application is expected to provide:",
        ),
        (
            "5.5 Research Gap Resolution",
            "The project addresses key research gaps identified in the literature survey:",
        ),
        (
            "5.6 Academic and Practical Contribution",
            "Beyond practical utility, this project serves as a comprehensive case study on "
            "applying hybrid deep learning and machine learning in medical image analysis. "
            "It demonstrates transfer learning, data augmentation, sklearn pipelines, Grad-CAM "
            "explainability, and Flask deployment in a single integrated system.",
        ),
    ]

    add_body(doc, outcomes[0][1])
    doc.add_paragraph()
    add_body(doc, outcomes[1][1])
    doc.add_paragraph()
    add_body(doc, outcomes[2][1])
    doc.add_paragraph()

    add_section_heading(doc, outcomes[3][0])
    web_features = [
        "Drag-and-drop MRI image upload interface.",
        "Instant prediction with class label and confidence percentage.",
        "Per-class probability bars for all four tumor categories.",
        "Side-by-side display of original MRI and Grad-CAM heatmap overlay.",
        "Responsive dark-theme UI compatible with desktop and mobile browsers.",
        "Clear error messages when artifacts are missing or upload fails.",
    ]
    for f in web_features:
        add_bullet(doc, f)

    add_section_heading(doc, outcomes[4][0])
    gaps = [
        "Kermany et al. (2018) gap: High GPU/data requirements for end-to-end CNN — resolved by hybrid CNN+SVM with frozen ResNet50.",
        "Deepak & Ameer (2019) gap: No visual explainability — resolved by Grad-CAM heatmap integration.",
        "Selvaraju et al. (2017) gap: Grad-CAM needs spatial CNN maps — resolved by dedicated CNN head model.",
        "Manual diagnosis gap: Slow and subjective radiologist review — resolved by automated instant classification.",
    ]
    for g in gaps:
        add_bullet(doc, g)

    doc.add_paragraph()
    add_body(doc, outcomes[5][1])


def build_references_and_sdg(doc):
    add_page_break(doc)
    add_heading(doc, "REFERENCES", level=1)
    references = [
        'Kermany, D. S., Goldbaum, M., Cai, W., et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," Cell, vol. 172, no. 5, pp. 1122-1131, 2018.',
        'S. Deepak and P. M. Ameer, "Brain Tumor Classification Using Deep CNN Features via Logistic Classifier," Journal of Medical Systems, vol. 43, no. 9, 2019.',
        'R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV, 2017.',
        'K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, pp. 770-778, 2016.',
        'Cortes, C. and Vapnik, V., "Support-Vector Networks," Machine Learning, vol. 20, no. 3, pp. 273-297, 1995.',
        'F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR, 2017.',
        'TensorFlow Documentation, "Keras Applications — ResNet50," https://keras.io/api/applications/resnet/, 2024.',
        'scikit-learn Documentation, "Support Vector Machines," https://scikit-learn.org/stable/modules/svm.html, 2024.',
        'Kaggle, "Brain Tumor MRI Dataset," https://www.kaggle.com/datasets, accessed 2025.',
        'S. J. Pan and Q. Yang, "A Survey on Transfer Learning," IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, 2010.',
    ]
    for i, ref in enumerate(references, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        run = p.add_run(f"[{i}] {ref}")
        set_run_font(run, size=11)

    add_page_break(doc)
    add_heading(doc, "SUSTAINABLE DEVELOPMENT GOAL (SDG) MAP", level=1)
    add_body(
        doc,
        "This project aligns with the United Nations Sustainable Development Goals (SDGs) "
        "by contributing to health innovation, responsible technology development, and "
        "accessible healthcare solutions.",
    )

    add_table(
        doc,
        ["SDG", "Target", "Core Project Contribution"],
        [
            (
                "SDG 3:\nGood Health\nand Well-Being",
                "Ensure healthy lives and promote well-being for all.",
                "Automated brain tumor detection from MRI scans supports early diagnosis "
                "and timely medical intervention, potentially improving patient outcomes "
                "and reducing diagnostic delays.",
            ),
            (
                "SDG 9:\nIndustry, Innovation\nand Infrastructure",
                "Build resilient infrastructure and foster innovation.",
                "Introduces a hybrid ML architecture combining ResNet50, SVM, and Grad-CAM "
                "— demonstrating innovation in medical AI using open-source technologies "
                "and transfer learning.",
            ),
            (
                "SDG 4:\nQuality Education",
                "Ensure inclusive and equitable quality education.",
                "Serves as a comprehensive academic project demonstrating deep learning, "
                "classical ML, Explainable AI, and web deployment — contributing to "
                "practical AI education in engineering curricula.",
            ),
            (
                "SDG 12:\nResponsible Consumption\nand Production",
                "Ensure sustainable consumption and production patterns.",
                "Built entirely on open-source technologies (Python, TensorFlow, Flask, "
                "scikit-learn) — promoting responsible, cost-effective, and sustainable "
                "software development without proprietary dependencies.",
            ),
        ],
        caption="Table 6.1: Sustainable Development Goals (SDG) Alignment Matrix",
    )


def build_document():
    doc = Document()
    apply_section_layout(doc.sections[0], MARGIN_COVER, header_footer=False)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    for style_name in ("Heading 1", "Heading 2", "Heading 3", "List Paragraph"):
        if style_name in doc.styles:
            doc.styles[style_name].font.name = "Times New Roman"

    build_title_pages(doc)
    build_abstract_and_toc(doc)
    build_chapter1(doc)
    build_chapter2(doc)
    build_chapter3(doc)
    build_chapter4(doc)
    build_chapter5(doc)
    build_references_and_sdg(doc)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Report saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    build_document()
