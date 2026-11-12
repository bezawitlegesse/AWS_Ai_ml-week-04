"""Creates data/matrix_basics.pdf, a short original study guide used as the sample document.
Run:  python scripts/make_sample_pdf.py   (needs the 'reportlab' package; the PDF is already included)."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("Matrix Basics: A Short Study Guide", "1. What is a matrix?", [
        "A matrix is a rectangular array of numbers arranged in rows and columns. The numbers are called the entries or elements of the matrix.",
        "A matrix with m rows and n columns has order m × n (read \"m by n\"). For example, a matrix with 2 rows and 3 columns has order 2 × 3. The entry in row i and column j is written a_ij.",
        "In this guide a matrix is written on one line with the rows separated by semicolons. For example, [1 2; 3 4] is the matrix whose first row is 1 2 and whose second row is 3 4.",
        "Matrices are used in engineering to describe systems of linear equations, electrical circuits, computer graphics and tables of data.",
    ]),
    (None, "2. Special types of matrices", [
        "A square matrix has the same number of rows and columns. A row matrix has only one row and a column matrix has only one column.",
        "A zero matrix has all of its entries equal to zero. A diagonal matrix is a square matrix whose entries outside the main diagonal are all zero.",
        "The identity matrix, written I, is a square matrix with 1 on the main diagonal and 0 everywhere else. Multiplying any matrix by the identity matrix of the right size leaves it unchanged: AI = IA = A.",
        "A symmetric matrix is a square matrix that is equal to its own transpose, that is A = A^T.",
    ]),
    (None, "3. Addition and scalar multiplication", [
        "Two matrices can be added only if they have the same order. To add them, add the corresponding entries. Subtraction works in the same way. For example, [1 2; 3 4] + [5 6; 7 8] = [6 8; 10 12].",
        "To multiply a matrix by a number (a scalar), multiply every entry by that number. For example, 3 × [1 2; 0 4] = [3 6; 0 12].",
        "Matrix addition is commutative, A + B = B + A, and associative, (A + B) + C = A + (B + C).",
    ]),
    (None, "4. Matrix multiplication", [
        "The product AB is defined only when the number of columns of A equals the number of rows of B. If A has order m × n and B has order n × p, then AB has order m × p.",
        "The entry in row i and column j of AB is found by multiplying the entries of row i of A by the matching entries of column j of B and adding the results.",
        "Example: if A = [1 2; 3 4] and B = [5 6; 7 8], then AB = [19 22; 43 50]. The first entry is 1×5 + 2×7 = 19.",
        "Matrix multiplication is generally not commutative. In most cases AB is not equal to BA. For the same A and B, BA = [23 34; 31 46], which is different from AB.",
    ]),
    (None, "5. Transpose and determinant", [
        "The transpose of a matrix A, written A^T, is obtained by swapping its rows and columns. The entry in row i and column j of A becomes the entry in row j and column i of A^T. If A has order m × n, then A^T has order n × m. For example, if A = [1 2 3; 4 5 6], then A^T = [1 4; 2 5; 3 6].",
        "The determinant of a 2 × 2 matrix [a b; c d] is ad - bc. For example, the determinant of [3 4; 2 5] is 3×5 - 4×2 = 7.",
        "A square matrix whose determinant is zero is called singular. A singular matrix has no inverse.",
    ]),
    (None, "6. The inverse of a matrix", [
        "The inverse of a square matrix A is written A^-1 and satisfies A A^-1 = A^-1 A = I. Only square matrices with a non-zero determinant have an inverse.",
        "For a 2 × 2 matrix [a b; c d], the inverse is 1/(ad - bc) times [d -b; -c a].",
        "Worked example: let A = [4 7; 2 6]. The determinant is 4×6 - 7×2 = 10. So A^-1 = 1/10 × [6 -7; -2 4] = [0.6 -0.7; -0.2 0.4].",
        "Inverses are used to solve systems of linear equations. If AX = B, then X = A^-1 B. A useful rule is (AB)^-1 = B^-1 A^-1.",
    ]),
]


def build(path):
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=22)
    heading = ParagraphStyle("h", parent=styles["Heading2"], fontSize=16, spaceAfter=10)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=12, leading=18, spaceAfter=10)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm,
                            title="Matrix Basics: A Short Study Guide", author="Sample document for the RAG project")
    story = []
    for index, (main_title, section, paragraphs) in enumerate(PAGES):
        if main_title:
            story += [Paragraph(main_title, title), Spacer(1, 12)]
        story.append(Paragraph(section, heading))
        story += [Paragraph(p, body) for p in paragraphs]
        if index < len(PAGES) - 1:
            story.append(PageBreak())
    doc.build(story)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "matrix_basics.pdf"
    build(out)
    print("Wrote", out)
