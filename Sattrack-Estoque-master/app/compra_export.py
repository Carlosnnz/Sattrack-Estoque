"""Exportação das compras exibidas em Movimentações."""

import io
from html import escape


HEADERS = (
    "ID", "Data do pedido", "Item", "Categoria", "Quantidade", "Custo unitário",
    "Custo total", "Fornecedor", "Filial/Setor", "Destinatário", "Registrado por", "Status",
)


def _safe_text(value):
    text = str(value or "")
    # Evita que dados cadastrados sejam interpretados como fórmulas no Excel.
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


def _row(compra):
    quantidade = compra.quantidade or 0
    custo = compra.custo_unitario or 0
    return (
        compra.id, _safe_text(compra.data_pedido), _safe_text(compra.item_nome),
        _safe_text(compra.categoria), quantidade, custo, quantidade * custo,
        _safe_text(compra.fornecedor), _safe_text(compra.destino),
        _safe_text(compra.destinatario), _safe_text(compra.usuario) or "Não informado",
        _safe_text(compra.status),
    )


def build_compras_xlsx(compras):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Compras"
    sheet.append(HEADERS)
    for compra in compras:
        sheet.append(_row(compra))
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = f"A1:L{sheet.max_row}"
    sheet.row_dimensions[1].height = 28
    widths = (9, 18, 35, 22, 14, 19, 18, 28, 25, 25, 25, 16)
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor="17365D")
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(vertical="center")
    for row in sheet.iter_rows(min_row=2):
        for column in (6, 7):
            row[column - 1].number_format = '"R$" #,##0.00'
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def build_compras_pdf(compras):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(A3), leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=28)
    header_style = ParagraphStyle("header", fontName="Helvetica-Bold", fontSize=8, textColor=colors.white, leading=10)
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=7, leading=9)
    title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=16, leading=20)
    labels = ("ID", "Data", "Item", "Categoria", "Qtd", "Custo un.", "Total", "Fornecedor", "Filial/Setor", "Destinatário", "Registrado por", "Status")
    rows = [[Paragraph(escape(label), header_style) for label in labels]]
    for compra in compras:
        values = _row(compra)
        display = list(values)
        for index in (5, 6):
            display[index] = f"R$ {values[index]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        rows.append([Paragraph(escape(str(value)), cell_style) for value in display])
    if not compras:
        rows.append([Paragraph("Nenhuma compra encontrada.", cell_style)] + [""] * 11)
    table = Table(rows, colWidths=(35, 66, 132, 85, 32, 62, 65, 110, 105, 100, 100, 65), repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    doc.build([Paragraph("Relatório de compras", title_style), Spacer(1, 12), table])
    return output.getvalue()
