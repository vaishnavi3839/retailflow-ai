"""
RetailFlow AI — PDF Report Generator
Produces a formatted executive summary from persisted footfall records.
"""

import os
import time
from datetime import datetime

from fpdf import FPDF
from database import SessionLocal, FootfallRecord


class _Report(FPDF):
    """Custom FPDF subclass with branded header and footer."""

    BRAND_R, BRAND_G, BRAND_B = 0, 180, 140  # Teal accent

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(self.BRAND_R, self.BRAND_G, self.BRAND_B)
        self.cell(0, 10, "RetailFlow AI", align="L")
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC", align="R")
        self.ln(4)
        self.set_draw_color(self.BRAND_R, self.BRAND_G, self.BRAND_B)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def create_pdf_report(output_dir: str = "reports") -> str:
    """
    Queries the last 20 persisted records and writes a PDF executive summary.

    Parameters
    ----------
    output_dir : directory to write the report into (created if absent)

    Returns
    -------
    Absolute path of the saved PDF file.
    """
    db = SessionLocal()
    try:
        records = (
            db.query(FootfallRecord)
            .order_by(FootfallRecord.timestamp.desc())
            .limit(20)
            .all()
        )
    finally:
        db.close()

    pdf = _Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title block ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 12, "Analytics Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Computer Vision Retail Analytics — Footfall Report", ln=True)
    pdf.ln(6)

    # ── Summary statistics ─────────────────────────────────────────────────
    if records:
        counts = [r.total_count for r in records]
        avg_count = sum(counts) / len(counts)
        peak_count = max(counts)
        top_zones: dict[str, int] = {}
        for r in records:
            top_zones[r.top_zone] = top_zones.get(r.top_zone, 0) + 1
        busiest_zone = max(top_zones, key=top_zones.get)

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, "Session Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(60, 7, "Records analysed:", ln=False)
        pdf.cell(0, 7, str(len(records)), ln=True)
        pdf.cell(60, 7, "Average footfall:", ln=False)
        pdf.cell(0, 7, f"{avg_count:.1f} persons", ln=True)
        pdf.cell(60, 7, "Peak footfall:", ln=False)
        pdf.cell(0, 7, f"{peak_count} persons", ln=True)
        pdf.cell(60, 7, "Most active zone:", ln=False)
        pdf.cell(0, 7, busiest_zone, ln=True)
        pdf.ln(6)

    # ── Data table ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(0, 180, 140)
    pdf.set_text_color(255, 255, 255)
    for header, width in [("Timestamp (UTC)", 75), ("Total Footfall", 55), ("Busiest Zone", 60)]:
        pdf.cell(width, 9, header, border=0, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    fill = False
    for row in records:
        pdf.set_fill_color(240, 248, 245) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(75, 8, row.timestamp.strftime("%Y-%m-%d  %H:%M:%S"), border="B", fill=True)
        pdf.cell(55, 8, str(row.total_count), border="B", fill=True, align="C")
        pdf.cell(60, 8, row.top_zone, border="B", fill=True)
        pdf.ln()
        fill = not fill

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"retailflow_report_{int(time.time())}.pdf")
    pdf.output(path)
    return os.path.abspath(path)


if __name__ == "__main__":
    saved = create_pdf_report()
    print(f"Report saved: {saved}")
