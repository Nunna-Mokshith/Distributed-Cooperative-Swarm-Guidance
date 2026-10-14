"""
generate_presentation.py
Generates a world-class, executive minimalist technical presentation in PowerPoint (.pptx)
Aesthetic: Light Warm Ivory background (#FAF7F2) with contrasting deep espresso (#1E130B)
           and saddle brown / warm cognac (#8B4513) accents.
Content: Purely technical Multi-Agent Swarm Guidance using Graph Attention Networks (GAT).
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ==============================================================================
# GLOBAL PALETTE & CONSTANTS
# ==============================================================================
COLOR_BG = RGBColor(250, 247, 242)          # #FAF7F2 Warm Ivory / Light Cream
COLOR_CARD = RGBColor(243, 236, 227)        # #F3ECE3 Warm Linen / Sand
COLOR_CARD_ALT = RGBColor(238, 229, 218)    # #EEE5DA Deeper Sand
COLOR_CARD_WHITE = RGBColor(255, 255, 255)  # #FFFFFF Crisp White
COLOR_BORDER = RGBColor(203, 187, 170)      # #CBBBAA Subtle Sand Border (1.2pt)
COLOR_BORDER_LIGHT = RGBColor(225, 215, 202)# #E1D7CA Light Inner Border
COLOR_PRIMARY = RGBColor(30, 19, 11)        # #1E130B Deep Rich Espresso Brown
COLOR_ACCENT = RGBColor(139, 69, 19)        # #8B4513 Saddle Brown / Warm Cognac
COLOR_ACCENT_MUTED = RGBColor(160, 82, 45)  # #A0522D Sienna / Terracotta
COLOR_SECONDARY = RGBColor(92, 78, 68)      # #5C4E44 Warm Slate Brown
COLOR_MUTED = RGBColor(138, 123, 110)       # #8A7B6E Soft Muted Brown
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_TAG_BG = RGBColor(234, 224, 212)      # #EAE0D4 Badge Fill
COLOR_CRITICAL_ACCENT = RGBColor(153, 51, 34) # #993322 Deep Rust for Attrition

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"
FONT_CODE = "Consolas"

# Initialize 16:9 widescreen presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]


# ==============================================================================
# REUSABLE GEOMETRIC & TYPOGRAPHY BUILDERS
# ==============================================================================
def apply_light_background(slide):
    """Draws full-bleed warm ivory background on slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG
    bg.line.color.rgb = COLOR_BG
    return bg


def add_header(slide, title_text, category="DECENTRALIZED MULTI-AGENT ARCHITECTURE", badge=None):
    """Adds minimal technical header with clean category hierarchy and accent divider."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.05))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.name = FONT_HEADING
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_ACCENT

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_PRIMARY
    p_title.space_after = Pt(4)

    # Subtle bottom divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.48), Inches(11.733), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_BORDER
    line.line.color.rgb = COLOR_BORDER

    if badge:
        b_width = Inches(1.8)
        b_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(12.533) - b_width, Inches(0.45), b_width, Inches(0.38))
        b_box.fill.solid()
        b_box.fill.fore_color.rgb = COLOR_TAG_BG
        b_box.line.color.rgb = COLOR_BORDER
        b_box.line.width = Pt(1.0)
        btf = b_box.text_frame
        btf.margin_left = btf.margin_top = btf.margin_right = btf.margin_bottom = 0
        bp = btf.paragraphs[0]
        bp.text = badge.upper()
        bp.alignment = PP_ALIGN.CENTER
        bp.font.name = FONT_HEADING
        bp.font.size = Pt(9)
        bp.font.bold = True
        bp.font.color.rgb = COLOR_ACCENT


def create_card(slide, left, top, width, height, bg_color=COLOR_CARD, border_color=COLOR_BORDER, border_width=Pt(1.2)):
    """Draws a clean structured card container with subtle borders."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = border_width
    return card


def add_stat_box(slide, left, top, width, height, value_str, label_str, sublabel_str="", bg_color=COLOR_CARD, value_color=COLOR_PRIMARY):
    """Draws a high-impact numerical callout card."""
    card = create_card(slide, left, top, width, height, bg_color=bg_color)
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.18)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.16)
    tf.margin_bottom = Inches(0.12)

    p_val = tf.paragraphs[0]
    p_val.text = value_str
    p_val.font.name = FONT_HEADING
    p_val.font.size = Pt(36)
    p_val.font.bold = True
    p_val.font.color.rgb = value_color

    p_lbl = tf.add_paragraph()
    p_lbl.text = label_str.upper()
    p_lbl.font.name = FONT_HEADING
    p_lbl.font.size = Pt(9.5)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = COLOR_ACCENT

    if sublabel_str:
        p_sub = tf.add_paragraph()
        p_sub.text = sublabel_str
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(8.5)
        p_sub.font.color.rgb = COLOR_SECONDARY
    return card


def add_formula_card(slide, left, top, width, height, title, formula_str, notes=[], definitions=[]):
    """Draws a structured mathematical formula card with distinct highlight box and symbol definitions."""
    container = create_card(slide, left, top, width, height, bg_color=COLOR_CARD)
    
    # Header tag
    tf_head = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.45)).text_frame
    tf_head.word_wrap = True
    tf_head.margin_left = tf_head.margin_top = tf_head.margin_right = tf_head.margin_bottom = 0
    p = tf_head.paragraphs[0]
    p.text = title.upper()
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    # Formula Box
    fbox_top = top + Inches(0.55)
    fbox_height = Inches(0.72)
    fbox = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(0.2), fbox_top, width - Inches(0.4), fbox_height)
    fbox.fill.solid()
    fbox.fill.fore_color.rgb = COLOR_CARD_WHITE
    fbox.line.color.rgb = COLOR_BORDER
    fbox.line.width = Pt(1.0)
    
    ftf = fbox.text_frame
    ftf.word_wrap = True
    ftf.margin_left = Inches(0.2)
    ftf.margin_right = Inches(0.2)
    ftf.margin_top = Inches(0.12)
    ftf.margin_bottom = Inches(0.08)
    fp = ftf.paragraphs[0]
    fp.text = formula_str
    fp.font.name = FONT_HEADING
    fp.font.size = Pt(13)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_PRIMARY

    curr_y = fbox_top + fbox_height + Inches(0.15)
    
    # Bullets / Notes
    if notes:
        n_box = slide.shapes.add_textbox(left + Inches(0.2), curr_y, width - Inches(0.4), Inches(0.9))
        ntf = n_box.text_frame
        ntf.word_wrap = True
        ntf.margin_left = ntf.margin_top = ntf.margin_right = ntf.margin_bottom = 0
        for i, note in enumerate(notes):
            np = ntf.paragraphs[0] if i == 0 else ntf.add_paragraph()
            np.text = f"•  {note}"
            np.font.name = FONT_BODY
            np.font.size = Pt(9.5)
            np.font.color.rgb = COLOR_SECONDARY
            np.space_after = Pt(4)
        curr_y += Inches(0.95)

    # Definitions grid
    if definitions:
        d_box = slide.shapes.add_textbox(left + Inches(0.2), curr_y, width - Inches(0.4), height - (curr_y - top) - Inches(0.12))
        dtf = d_box.text_frame
        dtf.word_wrap = True
        dtf.margin_left = dtf.margin_top = dtf.margin_right = dtf.margin_bottom = 0
        for i, (sym, desc) in enumerate(definitions):
            dp = dtf.paragraphs[0] if i == 0 else dtf.add_paragraph()
            dp.text = f"{sym}: {desc}"
            dp.font.name = FONT_BODY
            dp.font.size = Pt(9)
            dp.font.color.rgb = COLOR_SECONDARY
            dp.space_after = Pt(2)

    return container


# ==============================================================================
# SLIDE 1: MINIMALIST EXECUTIVE TITLE SLIDE
# ==============================================================================
slide1 = prs.slides.add_slide(blank_layout)
apply_light_background(slide1)

# Large Center Hero Card
hero = create_card(slide1, Inches(1.0), Inches(0.95), Inches(11.333), Inches(5.6), bg_color=COLOR_CARD, border_color=COLOR_BORDER, border_width=Pt(1.5))
# Top Accent Line on Hero Card
top_strip = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.95), Inches(11.333), Inches(0.12))
top_strip.fill.solid()
top_strip.fill.fore_color.rgb = COLOR_ACCENT
top_strip.line.color.rgb = COLOR_ACCENT

# Hero Text Container
hero_tf = slide1.shapes.add_textbox(Inches(1.6), Inches(1.4), Inches(10.133), Inches(2.7)).text_frame
hero_tf.word_wrap = True
hero_tf.margin_left = hero_tf.margin_top = hero_tf.margin_right = hero_tf.margin_bottom = 0

p = hero_tf.paragraphs[0]
p.text = "RESEARCH SPECIFICATION // MULTI-AGENT REINFORCEMENT & GRAPH ARCHITECTURES"
p.font.name = FONT_HEADING
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT

p = hero_tf.add_paragraph()
p.text = "Decentralized Swarm Target Allocation"
p.font.name = FONT_HEADING
p.font.size = Pt(36)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY
p.space_before = Pt(4)
p.space_after = Pt(6)

p = hero_tf.add_paragraph()
p.text = "Dynamic Graph Attention Network (GAT) with Size-Invariant Real-Time Consensus"
p.font.name = FONT_HEADING
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT_MUTED
p.space_after = Pt(14)

p = hero_tf.add_paragraph()
p.text = "Author: Nunna Mokshith | B.Tech Robotics & Artificial Intelligence\nTechnical Stack: PyTorch • Graph Attention (GAT) • Multi-Agent Optimization • Edge Kinematics"
p.font.name = FONT_BODY
p.font.size = Pt(11)
p.font.color.rgb = COLOR_SECONDARY

# 3 Highlight Pill Stat Boxes across bottom of Hero Card
stat_w = Inches(3.15)
stat_gap = Inches(0.34)
stat_left_start = Inches(1.6)
stat_top = Inches(4.35)
stat_h = Inches(1.65)

s1 = add_stat_box(slide1, stat_left_start, stat_top, stat_w, stat_h,
                  "1.10 ms", "GAT Decision Latency", "Deterministic edge CPU execution", bg_color=COLOR_CARD_WHITE)

s2 = add_stat_box(slide1, stat_left_start + stat_w + stat_gap, stat_top, stat_w, stat_h,
                  "O(N² + NM)", "Asymptotic Bound", "Size-invariant dynamic topology", bg_color=COLOR_CARD_WHITE)

s3 = add_stat_box(slide1, stat_left_start + (stat_w + stat_gap) * 2, stat_top, stat_w, stat_h,
                  "50 ms", "Attrition Failover", "Instant mid-flight consensus recovery", bg_color=COLOR_CARD_WHITE)


# ==============================================================================
# SLIDE 2: THE CORE PROBLEM: TARGET OVERKILL & JAMMING VULNERABILITY
# ==============================================================================
slide2 = prs.slides.add_slide(blank_layout)
apply_light_background(slide2)
add_header(slide2, "The Core Problem: Target Overkill & Centralized Jamming Vulnerability",
           category="OPERATIONAL VULNERABILITY ANALYSIS", badge="PARADIGM BREAKDOWN")

# Left Column: Greedy Seekers
col_w = Inches(5.7)
col_h = Inches(4.3)
c_left = Inches(0.8)
c_right = Inches(6.83)
top_y = Inches(1.68)

card_l2 = create_card(slide2, c_left, top_y, col_w, col_h)
tf2_l = card_l2.text_frame
tf2_l.word_wrap = True
tf2_l.margin_left = tf2_l.margin_right = Inches(0.3)
tf2_l.margin_top = tf2_l.margin_bottom = Inches(0.22)

p = tf2_l.paragraphs[0]
p.text = "PARADIGM FAILURE 1: GREEDY INDEPENDENT SEEKERS"
p.font.name = FONT_HEADING
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = tf2_l.add_paragraph()
p.text = "UNCOORDINATED HEURISTIC INTERCEPT"
p.font.name = FONT_HEADING
p.font.size = Pt(9.5)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(10)

# Stat Box Inside Left Card
callout_l2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.3), top_y + Inches(0.8), col_w - Inches(0.6), Inches(0.85))
callout_l2.fill.solid()
callout_l2.fill.fore_color.rgb = COLOR_CARD_WHITE
callout_l2.line.color.rgb = COLOR_BORDER
c_tf = callout_l2.text_frame
c_tf.word_wrap = True
c_tf.margin_left = c_tf.margin_right = Inches(0.18)
c_tf.margin_top = Inches(0.08)
cp = c_tf.paragraphs[0]
cp.text = "67% - 100% OVERKILL PROBABILITY"
cp.font.name = FONT_HEADING
cp.font.size = Pt(13)
cp.font.bold = True
cp.font.color.rgb = COLOR_CRITICAL_ACCENT
cp2 = c_tf.add_paragraph()
cp2.text = "Multiple interceptors collapse onto the same closest target, leaving threats unaddressed."
cp2.font.name = FONT_BODY
cp2.font.size = Pt(8.5)
cp2.font.color.rgb = COLOR_SECONDARY

# 3 Structured Insight Cards inside Left Column
flaws = [
    ("Uncoordinated argmin:", "Every agent optimizes min_j ||p_i - t_j|| in complete isolation."),
    ("Target Convergence:", "All missiles lock onto Target 1; Targets 2 & 3 penetrate defenses untouched."),
    ("Lethal Weapon Waste:", "4 interceptors expended on 1 hostile wastes 75% of swarm combat effectiveness.")
]
flaw_y = top_y + Inches(1.8)
for title_flaw, desc_flaw in flaws:
    f_box = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.3), flaw_y, col_w - Inches(0.6), Inches(0.68))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = COLOR_CARD_WHITE
    f_box.line.color.rgb = COLOR_BORDER_LIGHT
    f_box.line.width = Pt(0.8)
    ftf = f_box.text_frame
    ftf.word_wrap = True
    ftf.margin_left = ftf.margin_right = Inches(0.16)
    ftf.margin_top = Inches(0.08)
    fp = ftf.paragraphs[0]
    fp.text = title_flaw
    fp.font.name = FONT_HEADING
    fp.font.size = Pt(9.5)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_PRIMARY
    fp2 = ftf.add_paragraph()
    fp2.text = desc_flaw
    fp2.font.name = FONT_BODY
    fp2.font.size = Pt(8.5)
    fp2.font.color.rgb = COLOR_SECONDARY
    flaw_y += Inches(0.78)

# Right Column: Centralized C2 / Hungarian
card_r2 = create_card(slide2, c_right, top_y, col_w, col_h)
tf2_r = card_r2.text_frame
tf2_r.word_wrap = True
tf2_r.margin_left = tf2_r.margin_right = Inches(0.3)
tf2_r.margin_top = tf2_r.margin_bottom = Inches(0.22)

p = tf2_r.paragraphs[0]
p.text = "PARADIGM FAILURE 2: CENTRALIZED C2 / HUNGARIAN"
p.font.name = FONT_HEADING
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = tf2_r.add_paragraph()
p.text = "SINGLE-POINT-OF-FAILURE (SPOF) ARCHITECTURE"
p.font.name = FONT_HEADING
p.font.size = Pt(9.5)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(10)

# Stat Box Inside Right Card
callout_r2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_right + Inches(0.3), top_y + Inches(0.8), col_w - Inches(0.6), Inches(0.85))
callout_r2.fill.solid()
callout_r2.fill.fore_color.rgb = COLOR_CARD_WHITE
callout_r2.line.color.rgb = COLOR_BORDER
c_tf = callout_r2.text_frame
c_tf.word_wrap = True
c_tf.margin_left = c_tf.margin_right = Inches(0.18)
c_tf.margin_top = Inches(0.08)
cp = c_tf.paragraphs[0]
cp.text = "FATAL ELECTRONIC WARFARE (EW) VULNERABILITY"
cp.font.name = FONT_HEADING
cp.font.size = Pt(13)
cp.font.bold = True
cp.font.color.rgb = COLOR_CRITICAL_ACCENT
cp2 = c_tf.add_paragraph()
cp2.text = "Uplink jamming severs communication; a single jammed server blinds the entire swarm."
cp2.font.name = FONT_BODY
cp2.font.size = Pt(8.5)
cp2.font.color.rgb = COLOR_SECONDARY

c2_flaws = [
    ("Global State Uplink:", "Requires continuous telemetry upload X in R^(2N+2M) to mother craft or ground base."),
    ("RF Jamming Severance:", "Directional electronic jamming severs the uplink, rendering all missiles uncoordinated."),
    ("O(N³) Scalability Wall:", "Hungarian bipartite matching latency blows past the strict 50 ms flight control deadline.")
]
flaw_y = top_y + Inches(1.8)
for title_flaw, desc_flaw in c2_flaws:
    f_box = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_right + Inches(0.3), flaw_y, col_w - Inches(0.6), Inches(0.68))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = COLOR_CARD_WHITE
    f_box.line.color.rgb = COLOR_BORDER_LIGHT
    f_box.line.width = Pt(0.8)
    ftf = f_box.text_frame
    ftf.word_wrap = True
    ftf.margin_left = ftf.margin_right = Inches(0.16)
    ftf.margin_top = Inches(0.08)
    fp = ftf.paragraphs[0]
    fp.text = title_flaw
    fp.font.name = FONT_HEADING
    fp.font.size = Pt(9.5)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_PRIMARY
    fp2 = ftf.add_paragraph()
    fp2.text = desc_flaw
    fp2.font.name = FONT_BODY
    fp2.font.size = Pt(8.5)
    fp2.font.color.rgb = COLOR_SECONDARY
    flaw_y += Inches(0.78)

# Bottom Full-Width Mandate Banner
banner = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.12), Inches(11.733), Inches(0.88))
banner.fill.solid()
banner.fill.fore_color.rgb = COLOR_CARD_WHITE
banner.line.color.rgb = COLOR_ACCENT
banner.line.width = Pt(1.5)
btf = banner.text_frame
btf.word_wrap = True
btf.margin_left = Inches(0.3)
btf.margin_top = Inches(0.12)
bp1 = btf.paragraphs[0]
bp1.text = "THE DECENTRALIZED MANDATE: AD-HOC RF MESH & ZERO SERVER RELIANCE"
bp1.font.name = FONT_HEADING
bp1.font.size = Pt(11)
bp1.font.bold = True
bp1.font.color.rgb = COLOR_ACCENT
bp2 = btf.add_paragraph()
bp2.text = "To survive active combat EW jamming and eliminate target overkill, guidance MUST be fully distributed: zero centralized nodes, peer-to-peer ad-hoc consensus, and deterministic edge execution under 2.0 ms."
bp2.font.name = FONT_BODY
bp2.font.size = Pt(9.5)
bp2.font.color.rgb = COLOR_SECONDARY


# ==============================================================================
# SLIDE 3: GRAPH TOPOLOGY & PERMUTATION EQUIVARIANCE
# ==============================================================================
slide3 = prs.slides.add_slide(blank_layout)
apply_light_background(slide3)
add_header(slide3, "Graph Topology: Permutation Equivariance & Dynamic Size-Invariance",
           category="THEORETICAL FOUNDATION // GRAPH FORMULATION", badge="MATHEMATICAL RIGOR")

# Left Column: Standard MLPs Failure
card_l3 = create_card(slide3, c_left, top_y, col_w, Inches(5.3))
tf3_l = card_l3.text_frame
tf3_l.word_wrap = True
tf3_l.margin_left = tf3_l.margin_right = Inches(0.3)
tf3_l.margin_top = tf3_l.margin_bottom = Inches(0.2)

p = tf3_l.paragraphs[0]
p.text = "THE RIGIDITY BOTTLENECK: STANDARD MLPs"
p.font.name = FONT_HEADING
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = tf3_l.add_paragraph()
p.text = "WHY DENSE MULTI-LAYER PERCEPTRONS FAIL IN TACTICAL SWARMS"
p.font.name = FONT_HEADING
p.font.size = Pt(9)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(8)

# Formula box on MLP side
mlp_box = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.3), top_y + Inches(0.72), col_w - Inches(0.6), Inches(0.75))
mlp_box.fill.solid()
mlp_box.fill.fore_color.rgb = COLOR_CARD_WHITE
mlp_box.line.color.rgb = COLOR_BORDER
mtf = mlp_box.text_frame
mtf.word_wrap = True
mtf.margin_left = Inches(0.18)
mtf.margin_top = Inches(0.1)
mp = mtf.paragraphs[0]
mp.text = "Fixed Input Tensor:  f : R^(N x d)  -->  R^N"
mp.font.name = FONT_CODE
mp.font.size = Pt(12)
mp.font.bold = True
mp.font.color.rgb = COLOR_PRIMARY
mp2 = mtf.add_paragraph()
mp2.text = "Rigid weight matrices W_dense in R^(d_hidden x N·d) bind network to fixed cardinality N."
mp2.font.name = FONT_BODY
mp2.font.size = Pt(8.5)
mp2.font.color.rgb = COLOR_SECONDARY

mlp_flaws = [
    ("Hardcoded Cardinality N:", "Dense layers fix the exact number of agents at compile time. A network trained for N=4 cannot accept 5 or 6 drones without throwing dimension errors."),
    ("Mid-Air Attrition Crash:", "When drone a_0 is intercepted, active agent count drops N -> N-1. An MLP immediately suffers tensor shape mismatch exceptions during mid-flight guidance."),
    ("Combinatorial Order Bias:", "Permuting agent order in the input vector produces wildly different allocation outputs unless trained with prohibitive N! factorial data permutations.")
]
flaw_y3 = top_y + Inches(1.6)
for title_flaw, desc_flaw in mlp_flaws:
    f_box = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.3), flaw_y3, col_w - Inches(0.6), Inches(1.1))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = COLOR_CARD_WHITE
    f_box.line.color.rgb = COLOR_BORDER_LIGHT
    f_box.line.width = Pt(0.8)
    ftf = f_box.text_frame
    ftf.word_wrap = True
    ftf.margin_left = ftf.margin_right = Inches(0.16)
    ftf.margin_top = Inches(0.1)
    fp = ftf.paragraphs[0]
    fp.text = title_flaw
    fp.font.name = FONT_HEADING
    fp.font.size = Pt(10)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_PRIMARY
    fp2 = ftf.add_paragraph()
    fp2.text = desc_flaw
    fp2.font.name = FONT_BODY
    fp2.font.size = Pt(8.5)
    fp2.font.color.rgb = COLOR_SECONDARY
    flaw_y3 += Inches(1.2)

# Right Column: Graph Neural Network Invariance
card_r3 = create_card(slide3, c_right, top_y, col_w, Inches(5.3))
tf3_r = card_r3.text_frame
tf3_r.word_wrap = True
tf3_r.margin_left = tf3_r.margin_right = Inches(0.3)
tf3_r.margin_top = tf3_r.margin_bottom = Inches(0.2)

p = tf3_r.paragraphs[0]
p.text = "THE SOLUTION: GRAPH EQUIVARIANT INVARIANCE"
p.font.name = FONT_HEADING
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = tf3_r.add_paragraph()
p.text = "SIZE-INVARIANT NEIGHBORHOOD MESSAGE PASSING"
p.font.name = FONT_HEADING
p.font.size = Pt(9)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(8)

# Formula box on GNN side
gnn_box = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_right + Inches(0.3), top_y + Inches(0.72), col_w - Inches(0.6), Inches(0.75))
gnn_box.fill.solid()
gnn_box.fill.fore_color.rgb = COLOR_CARD_WHITE
gnn_box.line.color.rgb = COLOR_BORDER
gtf = gnn_box.text_frame
gtf.word_wrap = True
gtf.margin_left = Inches(0.18)
gtf.margin_top = Inches(0.1)
gp = gtf.paragraphs[0]
gp.text = "Equivariance:  f( P X ,  P A P^T )  =  P f( X , A )"
gp.font.name = FONT_CODE
gp.font.size = Pt(12)
gp.font.bold = True
gp.font.color.rgb = COLOR_PRIMARY
gp2 = gtf.add_paragraph()
gp2.text = "For all permutation matrices P in {0, 1}^(N x N). Internal memory ordering never alters tactical decisions."
gp2.font.name = FONT_BODY
gp2.font.size = Pt(8.5)
gp2.font.color.rgb = COLOR_SECONDARY

gnn_strengths = [
    ("Dynamic Adjacency A(t):", "A_ij(t) = 1 if ||p_i - p_j|| <= R_comm (12.0 km), else 0. Naturally handles RF range limits, ad-hoc link formation, and enemy jamming packet drops."),
    ("Local Neighborhood Sets N_i:", "Convolutional message passing sums over immediate peers N_i rather than global vector indices. Cardinality N dynamically expands or contracts in real time."),
    ("Zero Retraining Generalization:", "Trained network weights deploy out-of-the-box onto arbitrary swarm sizes (N=3, 4, 8, or 20) with mathematically guaranteed shape stability.")
]
flaw_y3 = top_y + Inches(1.6)
for title_flaw, desc_flaw in gnn_strengths:
    f_box = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_right + Inches(0.3), flaw_y3, col_w - Inches(0.6), Inches(1.1))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = COLOR_CARD_WHITE
    f_box.line.color.rgb = COLOR_BORDER_LIGHT
    f_box.line.width = Pt(0.8)
    ftf = f_box.text_frame
    ftf.word_wrap = True
    ftf.margin_left = ftf.margin_right = Inches(0.16)
    ftf.margin_top = Inches(0.1)
    fp = ftf.paragraphs[0]
    fp.text = title_flaw
    fp.font.name = FONT_HEADING
    fp.font.size = Pt(10)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_PRIMARY
    fp2 = ftf.add_paragraph()
    fp2.text = desc_flaw
    fp2.font.name = FONT_BODY
    fp2.font.size = Pt(8.5)
    fp2.font.color.rgb = COLOR_SECONDARY
    flaw_y3 += Inches(1.2)


# ==============================================================================
# SLIDE 4: NEURAL ARCHITECTURE: DECOMPOSED GRAPH ATTENTION (GAT) LAYER
# ==============================================================================
slide4 = prs.slides.add_slide(blank_layout)
apply_light_background(slide4)
add_header(slide4, "Neural Architecture: Decomposed Graph Attention (GAT) Layer",
           category="DEEP LEARNING ARCHITECTURE SPECIFICATION", badge="EDGE PYTORCH CORE")

# Top: Visual Step-by-Step Pipeline Blocks (4 blocks)
pipe_w = Inches(2.76)
pipe_h = Inches(1.42)
pipe_gap = Inches(0.23)
pipe_y = Inches(1.65)

pipeline_steps = [
    ("STEP 01", "Kinematic Ingestion", "Relative 4D state vector:\n[ (p_i - p_j)/S, (v_i - v_j)/S ]\nNormalized by S = 10.0 km"),
    ("STEP 02", "Linear Latent Projection", "Shared linear projection:\nW in R^(16 x 4) transforms raw\nkinematics into 16D latent space"),
    ("STEP 03", "Decomposed Attention", "Directional projection vectors:\ne_ij = LeakyReLU(W h_i · a_src\n              + W h_j · a_dst)"),
    ("STEP 04", "Message Aggregation", "Softmax-weighted peer sum:\nh'_i = ELU( Σ_{j in N_i} α_ij · W h_j )\nOutputs 16D refined node context")
]

for idx, (step_num, step_name, step_detail) in enumerate(pipeline_steps):
    step_left = Inches(0.8) + idx * (pipe_w + pipe_gap)
    p_card = create_card(slide4, step_left, pipe_y, pipe_w, pipe_h, bg_color=COLOR_CARD)
    ptf = p_card.text_frame
    ptf.word_wrap = True
    ptf.margin_left = ptf.margin_right = Inches(0.14)
    ptf.margin_top = Inches(0.1)
    
    pp1 = ptf.paragraphs[0]
    pp1.text = step_num
    pp1.font.name = FONT_HEADING
    pp1.font.size = Pt(8.5)
    pp1.font.bold = True
    pp1.font.color.rgb = COLOR_ACCENT

    pp2 = ptf.add_paragraph()
    pp2.text = step_name
    pp2.font.name = FONT_HEADING
    pp2.font.size = Pt(11)
    pp2.font.bold = True
    pp2.font.color.rgb = COLOR_PRIMARY
    pp2.space_after = Pt(2)

    pp3 = ptf.add_paragraph()
    pp3.text = step_detail
    pp3.font.name = FONT_CODE
    pp3.font.size = Pt(7.5)
    pp3.font.color.rgb = COLOR_SECONDARY

# Bottom Left: Dedicated Formula Highlight Card
b_left_w = Inches(6.3)
b_right_w = Inches(5.1)
bot_y = Inches(3.25)
bot_h = Inches(3.75)

add_formula_card(
    slide4, Inches(0.8), bot_y, b_left_w, bot_h,
    title="DECOMPOSED GRAPH ATTENTION MATHEMATICAL SPECIFICATION",
    formula_str="α_ij = exp( e_ij ) / Σ_{k in N_i U {i}} exp( e_ik )\ne_ij = LeakyReLU( W h_i · a_src + W h_j · a_dst )",
    notes=[
        "Masked Softmax Normalization ensures Σ_j α_ij = 1.0 across active peer links.",
        "Non-linear activation LeakyReLU with negative slope 0.2 retains negative gradient flow.",
        "Aggregation layer: h'_i = ELU( Σ_{j in N_i} α_ij W h_j ) produces smooth contextual representations."
    ],
    definitions=[
        ("W in R^(16 x 4)", "Shared linear transformation matrix"),
        ("a_src, a_dst in R^16", "Directional orthogonal attention projection vectors"),
        ("N_i", "Ad-hoc communication neighborhood (||p_i - p_j|| <= 12.0 km)"),
        ("α_ij in (0, 1)", "Dynamic learned affinity coefficient between drone i and drone j")
    ]
)

# Bottom Right: Tactical Significance & Computational Efficiency Card
b_card_r = create_card(slide4, Inches(7.433), bot_y, b_right_w, bot_h)
btf_r = b_card_r.text_frame
btf_r.word_wrap = True
btf_r.margin_left = btf_r.margin_right = Inches(0.25)
btf_r.margin_top = Inches(0.2)

p = btf_r.paragraphs[0]
p.text = "TACTICAL ADVANTAGES & COMPUTE EFFICIENCY"
p.font.name = FONT_HEADING
p.font.size = Pt(11.5)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = btf_r.add_paragraph()
p.text = "HARDWARE-IN-THE-LOOP EDGE ADVANTAGES"
p.font.name = FONT_HEADING
p.font.size = Pt(9)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(10)

advs = [
    ("Decomposed Dot-Product Optimization:", "Computing W h_i · a once per node reduces projection overhead from O(|E|) to O(|V|), cutting edge memory transfers by 62%."),
    ("Sub-Millisecond Inference (< 0.40 ms):", "Pure PyTorch vectorization executes the entire forward pass in 0.38 ms on standard edge CPUs, comfortably fitting in 20 Hz (50 ms) flight guidance cycles."),
    ("Analytical Zero-Training Initialization:", "Orthogonal weight initialization provides balanced attention distribution out of the box, completely eliminating Multi-Agent PPO training collapse.")
]
adv_y = bot_y + Inches(0.7)
for t_adv, d_adv in advs:
    a_box = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.433) + Inches(0.2), adv_y, b_right_w - Inches(0.4), Inches(0.88))
    a_box.fill.solid()
    a_box.fill.fore_color.rgb = COLOR_CARD_WHITE
    a_box.line.color.rgb = COLOR_BORDER_LIGHT
    a_box.line.width = Pt(0.8)
    atf = a_box.text_frame
    atf.word_wrap = True
    atf.margin_left = atf.margin_right = Inches(0.16)
    atf.margin_top = Inches(0.08)
    ap = atf.paragraphs[0]
    ap.text = t_adv
    ap.font.name = FONT_HEADING
    ap.font.size = Pt(9.5)
    ap.font.bold = True
    ap.font.color.rgb = COLOR_PRIMARY
    ap2 = atf.add_paragraph()
    ap2.text = d_adv
    ap2.font.name = FONT_BODY
    ap2.font.size = Pt(8.5)
    ap2.font.color.rgb = COLOR_SECONDARY
    adv_y += Inches(0.96)


# ==============================================================================
# SLIDE 5: CONFLICT RESOLUTION: TACTICAL UTILITY & MUTUAL INHIBITION
# ==============================================================================
slide5 = prs.slides.add_slide(blank_layout)
apply_light_background(slide5)
add_header(slide5, "Conflict Resolution: Tactical Utility & Cooperative Mutual Inhibition",
           category="MULTI-AGENT CONSENSUS ALGORITHM", badge="ANTI-OVERKILL LOGIC")

# Left Column: Kinematic Utility Formulation
card_l5 = create_card(slide5, c_left, top_y, col_w, Inches(5.3))
tf5_l = card_l5.text_frame
tf5_l.word_wrap = True
tf5_l.margin_left = tf5_l.margin_right = Inches(0.25)
tf5_l.margin_top = Inches(0.2)

p = tf5_l.paragraphs[0]
p.text = "1. KINEMATIC TACTICAL UTILITY MATRIX"
p.font.name = FONT_HEADING
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = tf5_l.add_paragraph()
p.text = "INDIVIDUAL UTILITY MATRIX:  U in R^(N x M)"
p.font.name = FONT_HEADING
p.font.size = Pt(9)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(8)

# Formula Box
u_box = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.25), top_y + Inches(0.72), col_w - Inches(0.5), Inches(0.95))
u_box.fill.solid()
u_box.fill.fore_color.rgb = COLOR_CARD_WHITE
u_box.line.color.rgb = COLOR_BORDER
utf = u_box.text_frame
utf.word_wrap = True
utf.margin_left = Inches(0.18)
utf.margin_top = Inches(0.12)
up1 = utf.paragraphs[0]
up1.text = "U_ij = 5.0 · [ 1 / ( d_ij / 1000 + 1 ) ]"
up1.font.name = FONT_CODE
up1.font.size = Pt(13)
up1.font.bold = True
up1.font.color.rgb = COLOR_PRIMARY
up2 = utf.add_paragraph()
up2.text = "Pure Geometric Tactical Utility: Evaluates target proximity with zero artificial priority bias."
up2.font.name = FONT_BODY
up2.font.size = Pt(8.5)
up2.font.color.rgb = COLOR_SECONDARY

u_breakdown = [
    ("Hyperbolic Range Decay:  5.0 · [ 1 / (d/1000 + 1) ]",
     "• Monotonically decreases as Euclidean distance d_ij expands.\n"
     "• At d = 0 km (point-blank), range value = 5.0.\n"
     "• At d = 4 km, range value = 1.0; at d = 10 km, range value = 0.45.\n"
     "• Treats all targets as equal threats that must be destroyed based on proximity."),
    ("Fail-Safe Self-Destruct & Surplus Protocol",
     "• 1-to-1 matching: each threat is guaranteed a dedicated interceptor.\n"
     "• If all targets are neutralized, excess interceptors immediately self-destruct.\n"
     "• Prevents uncontrolled kinetic crashes or wandering off-course.\n"
     "• Operates autonomously without requiring human intervention.")
]
ub_y = top_y + Inches(1.82)
for title_ub, desc_ub in u_breakdown:
    b_box = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.25), ub_y, col_w - Inches(0.5), Inches(1.5))
    b_box.fill.solid()
    b_box.fill.fore_color.rgb = COLOR_CARD_WHITE
    b_box.line.color.rgb = COLOR_BORDER_LIGHT
    b_box.line.width = Pt(0.8)
    btf = b_box.text_frame
    btf.word_wrap = True
    btf.margin_left = btf.margin_right = Inches(0.16)
    btf.margin_top = Inches(0.1)
    bp = btf.paragraphs[0]
    bp.text = title_ub
    bp.font.name = FONT_HEADING
    bp.font.size = Pt(9.5)
    bp.font.bold = True
    bp.font.color.rgb = COLOR_PRIMARY
    bp2 = btf.add_paragraph()
    bp2.text = desc_ub
    bp2.font.name = FONT_BODY
    bp2.font.size = Pt(8.5)
    bp2.font.color.rgb = COLOR_SECONDARY
    ub_y += Inches(1.6)

# Right Column: Cooperative Mutual Inhibition
card_r5 = create_card(slide5, c_right, top_y, col_w, Inches(5.3))
tf5_r = card_r5.text_frame
tf5_r.word_wrap = True
tf5_r.margin_left = tf5_r.margin_right = Inches(0.25)
tf5_r.margin_top = Inches(0.2)

p = tf5_r.paragraphs[0]
p.text = "2. COOPERATIVE MUTUAL INHIBITION"
p.font.name = FONT_HEADING
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = tf5_r.add_paragraph()
p.text = "ANTI-OVERKILL CONSENSUS EQUATION:  U' in R^(N x M)"
p.font.name = FONT_HEADING
p.font.size = Pt(9)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(8)

# Formula Box
inh_box = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_right + Inches(0.25), top_y + Inches(0.72), col_w - Inches(0.5), Inches(0.95))
inh_box.fill.solid()
inh_box.fill.fore_color.rgb = COLOR_CARD_WHITE
inh_box.line.color.rgb = COLOR_BORDER
itf = inh_box.text_frame
itf.word_wrap = True
itf.margin_left = Inches(0.18)
itf.margin_top = Inches(0.12)
ip1 = itf.paragraphs[0]
ip1.text = "U'_ij = U_ij - β · Σ_{k != i} α_ik · II( argmax_m U_km = j )"
ip1.font.name = FONT_CODE
ip1.font.size = Pt(12)
ip1.font.bold = True
ip1.font.color.rgb = COLOR_PRIMARY
ip2 = itf.add_paragraph()
ip2.text = "Penalizes redundant target commitment based on dynamic peer attention weights α_ik."
ip2.font.name = FONT_BODY
ip2.font.size = Pt(8.5)
ip2.font.color.rgb = COLOR_SECONDARY

inh_breakdown = [
    ("Peer Inhibitory Penalty:  β · α_ik  (β = 1.8)",
     "• If peer drone k in N_i already exhibits max utility for target j, an inhibitory penalty β · α_ik is subtracted from drone i's utility for target j.\n"
     "• The penalty is proportional to GAT attention α_ik: closer/more coordinated peers exert stronger inhibitory pressure.\n"
     "• Naturally forces drone i to diverge towards the next best unassigned target."),
    ("Softmax Bipartite Extraction:  P_ij ∝ exp( 2 · U'_ij )",
     "• Sharp temperature scaling converts penalized utilities into discrete probabilities.\n"
     "• Greedy bipartite extraction processes sorted tuples (P_ij, i, j) sequentially.\n"
     "• Provably guarantees conflict-free allocation in O(NM log(NM)) without iterative auction chatter or centralized C2 relays.")
]
ib_y = top_y + Inches(1.82)
for title_ib, desc_ib in inh_breakdown:
    b_box = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_right + Inches(0.25), ib_y, col_w - Inches(0.5), Inches(1.5))
    b_box.fill.solid()
    b_box.fill.fore_color.rgb = COLOR_CARD_WHITE
    b_box.line.color.rgb = COLOR_BORDER_LIGHT
    b_box.line.width = Pt(0.8)
    btf = b_box.text_frame
    btf.word_wrap = True
    btf.margin_left = btf.margin_right = Inches(0.16)
    btf.margin_top = Inches(0.1)
    bp = btf.paragraphs[0]
    bp.text = title_ib
    bp.font.name = FONT_HEADING
    bp.font.size = Pt(9.5)
    bp.font.bold = True
    bp.font.color.rgb = COLOR_PRIMARY
    bp2 = btf.add_paragraph()
    bp2.text = desc_ib
    bp2.font.name = FONT_BODY
    bp2.font.size = Pt(8.5)
    bp2.font.color.rgb = COLOR_SECONDARY
    ib_y += Inches(1.6)


# ==============================================================================
# SLIDE 6: COMBAT ATTRITION EXECUTION TRACE
# ==============================================================================
slide6 = prs.slides.add_slide(blank_layout)
apply_light_background(slide6)
add_header(slide6, "Combat Attrition Execution Trace: 4 Drones vs. 3 Hostile Targets",
           category="DYNAMIC RESILIENCE & ATTRITION TELEMETRY", badge="MID-AIR CASUALTY ADAPTATION")

timeline_stages = [
    {
        "time": "t = 0.00s",
        "title": "SYSTEM LAUNCH & INITIAL TOPOLOGY DISCOVERY",
        "badge": "COMPLETE MESH (K4)",
        "badge_color": COLOR_ACCENT,
        "bg_color": COLOR_CARD,
        "border_color": COLOR_BORDER,
        "points": [
            "Swarm deployment: 4 Interceptors A = { a_0, a_1, a_2, a_3 }, speed v = 450 m/s.",
            "Hostile Target Salvo: T = { t_0, t_1, t_2 } at 10.0 km range (All targets treated as equal threats).",
            "Complete ad-hoc mesh initialized: all pairwise distances ||p_i - p_j|| < 12.0 km R_comm."
        ]
    },
    {
        "time": "t = 0.05s",
        "title": "INITIAL GAT CONSENSUS & 1-TO-1 TARGET ALLOCATION",
        "badge": "LATENCY: 1.08 ms",
        "badge_color": COLOR_ACCENT,
        "bg_color": COLOR_CARD,
        "border_color": COLOR_BORDER,
        "points": [
            "Decomposed GAT forward pass completes in 1.08 ms across all agents.",
            "1-to-1 Allocation: a_0 -> t_0, a_1 -> t_1, a_2 -> t_2 (Every threat is covered).",
            "Surplus Management: Agent a_3 deployed in reserve/backup mode to eliminate unassigned leaks."
        ]
    },
    {
        "time": "t = 3.50s - 3.55s",
        "title": "MID-FLIGHT CASUALTY & REAL-TIME TOPOLOGY ADAPTATION",
        "badge": "CRITICAL ATTRITION FAILOVER (50 ms)",
        "badge_color": COLOR_CRITICAL_ACCENT,
        "bg_color": COLOR_CARD_WHITE,
        "border_color": COLOR_CRITICAL_ACCENT,
        "points": [
            "Casualty Event (t = 3.50s): Lead interceptor a_0 destroyed mid-air! Active graph shrinks: V(t) = { a_1, a_2, a_3 }.",
            "Dynamic Size Invariance: GAT evaluates contracted 3x3 adjacency matrix with zero dimension mismatch exceptions.",
            "Autonomous Re-Targeting (t = 3.55s, Δt = 50 ms): Agent a_3 detects t_0 uncovered and immediately promotes to primary tracker!",
            "100% Target Coverage Maintained: a_1 -> t_1, a_2 -> t_2, a_3 -> t_0 without centralized intervention."
        ]
    },
    {
        "time": "t = 22.10s",
        "title": "FINAL TERMINAL INTERCEPT & THREAT NEUTRALIZATION",
        "badge": "MISSION SUCCESS: 100%",
        "badge_color": COLOR_ACCENT,
        "bg_color": COLOR_CARD,
        "border_color": COLOR_BORDER,
        "points": [
            "All 3 hostile incoming threats intercepted and neutralized: a_3 hits t_0, a_1 hits t_1, a_2 hits t_2.",
            "Fail-Safe Protocol: Any surplus/excess interceptors safely execute controlled self-destruct.",
            "Zero redundant warheads or unguided crashes: Overkill rate strictly 0.0% across entire engagement."
        ]
    }
]

t_top = Inches(1.68)
t_h = Inches(1.22)
t_gap = Inches(0.12)

for stage in timeline_stages:
    sc = create_card(slide6, Inches(0.8), t_top, Inches(11.733), t_h, bg_color=stage["bg_color"], border_color=stage["border_color"])
    stf = sc.text_frame
    stf.word_wrap = True
    stf.margin_left = Inches(0.25)
    stf.margin_right = Inches(0.25)
    stf.margin_top = Inches(0.1)

    p0 = stf.paragraphs[0]
    p0.text = stage["time"] + "  |  " + stage["title"]
    p0.font.name = FONT_HEADING
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = stage["badge_color"]

    for pt in stage["points"]:
        p_pt = stf.add_paragraph()
        p_pt.text = "•  " + pt
        p_pt.font.name = FONT_BODY
        p_pt.font.size = Pt(8.8)
        p_pt.font.color.rgb = COLOR_PRIMARY if stage["bg_color"] == COLOR_CARD_WHITE else COLOR_SECONDARY
        p_pt.space_after = Pt(1)

    t_top += t_h + t_gap


# ==============================================================================
# SLIDE 7: EMPIRICAL TELEMETRY & BENCHMARK DASHBOARD
# ==============================================================================
slide7 = prs.slides.add_slide(blank_layout)
apply_light_background(slide7)
add_header(slide7, "Empirical Telemetry & Real-Time Performance Dashboard",
           category="SIMULATION VERIFICATION // TELEMETRY ANALYSIS", badge="HARDWARE BENCHMARK")

# Embed Plot Image
img_w = Inches(7.8)
img_h = Inches(5.3)
img_left = Inches(0.8)
img_top = Inches(1.68)

plot_path = os.path.join("figures", "live_engagement_telemetry.png")
if os.path.exists(plot_path):
    # Border container for the plot
    img_card = create_card(slide7, img_left, img_top, img_w, img_h, bg_color=COLOR_CARD_WHITE)
    slide7.shapes.add_picture(plot_path, img_left + Inches(0.1), img_top + Inches(0.1), img_w - Inches(0.2), img_h - Inches(0.2))
else:
    # Fallback container
    fb = create_card(slide7, img_left, img_top, img_w, img_h, bg_color=COLOR_CARD_WHITE)
    ftf = fb.text_frame
    ftf.word_wrap = True
    fp = ftf.paragraphs[0]
    fp.text = "TELEMETRY PLOT: figures/live_engagement_telemetry.png"
    fp.font.name = FONT_HEADING
    fp.font.size = Pt(14)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_PRIMARY

# Right Column: 4 Big Stat Callouts Stacked Vertically
stat_r_left = Inches(8.85)
stat_r_w = Inches(3.683)
stat_r_top = Inches(1.68)
stat_box_h = Inches(1.22)
stat_box_gap = Inches(0.14)

stats_data = [
    ("100.0%", "INTERCEPT SUCCESS RATE", "3 of 3 Hostile Threats Neutralized", COLOR_PRIMARY),
    ("0.0%", "TARGET OVERKILL RATE", "Zero Redundant Warhead Expenditure", COLOR_ACCENT),
    ("1.10 ms", "GAT DECISION LATENCY", "Deterministic Edge CPU Execution", COLOR_PRIMARY),
    ("12.0 km", "AD-HOC MESH RANGE", "Autonomous Peer Link Discovery", COLOR_ACCENT)
]

for idx, (val, lbl, sub, val_c) in enumerate(stats_data):
    s_top = stat_r_top + idx * (stat_box_h + stat_box_gap)
    add_stat_box(slide7, stat_r_left, s_top, stat_r_w, stat_box_h,
                 val, lbl, sub, bg_color=COLOR_CARD, value_color=val_c)


# ==============================================================================
# SLIDE 8: COMPUTATIONAL COMPLEXITY & SYSTEM DEFENSE
# ==============================================================================
slide8 = prs.slides.add_slide(blank_layout)
apply_light_background(slide8)
add_header(slide8, "Technical Defense: Computational Complexity & Jamming Resilience",
           category="SYSTEM RIGOR // TECHNICAL DEFENSE", badge="COMPLEXITY VERIFIED")

# Left Column: Comparative Table
tbl_left = Inches(0.8)
tbl_top = Inches(1.68)
tbl_w = Inches(7.5)
tbl_h = Inches(5.3)

table_shape = slide8.shapes.add_table(7, 4, tbl_left, tbl_top, tbl_w, tbl_h)
tbl = table_shape.table

# Column widths
tbl.columns[0].width = Inches(2.2)
tbl.columns[1].width = Inches(1.7)
tbl.columns[2].width = Inches(1.7)
tbl.columns[3].width = Inches(1.9)

headers = ["Evaluation Metric", "Centralized (C2)", "Greedy Seeker", "Our GAT Swarm"]
table_rows = [
    ("Time Complexity", "O(N³)", "O(NM)", "O(N² + NM log(NM))"),
    ("Communication Link", "Global Uplink", "Zero (Blind)", "Ad-Hoc Peer RF Mesh"),
    ("Single Point of Failure", "High (Mother Craft)", "None (Uncoordinated)", "Zero (Fully Distributed)"),
    ("EW Jamming Resilience", "Fatal System Crash", "Unaffected", "High (Masked Graph)"),
    ("Attrition Failover", "Slow Re-upload", "Target Leakage", "Instantaneous (≤ 50 ms)"),
    ("Overkill Waste Rate", "0.0% (If unjammed)", "67% - 100% (High)", "0.0% (Guaranteed)")
]

# Style Header Row
for c_idx, head_text in enumerate(headers):
    cell = tbl.cell(0, c_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = COLOR_PRIMARY
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    ctf = cell.text_frame
    ctf.word_wrap = True
    ctf.margin_left = ctf.margin_right = Inches(0.12)
    ctf.margin_top = ctf.margin_bottom = Inches(0.06)
    cp = ctf.paragraphs[0]
    cp.text = head_text
    cp.font.name = FONT_HEADING
    cp.font.size = Pt(10)
    cp.font.bold = True
    cp.font.color.rgb = COLOR_WHITE
    if c_idx > 0:
        cp.alignment = PP_ALIGN.CENTER

# Style Data Rows
for r_idx, row_vals in enumerate(table_rows):
    bg_row = COLOR_CARD_WHITE if r_idx % 2 == 0 else COLOR_CARD
    for c_idx, val in enumerate(row_vals):
        cell = tbl.cell(r_idx + 1, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_row
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        ctf = cell.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_right = Inches(0.12)
        ctf.margin_top = ctf.margin_bottom = Inches(0.06)
        cp = ctf.paragraphs[0]
        cp.text = val
        cp.font.name = FONT_BODY
        cp.font.size = Pt(9.5)
        if c_idx == 0:
            cp.font.bold = True
            cp.font.color.rgb = COLOR_PRIMARY
        elif c_idx == 3:
            cp.font.bold = True
            cp.font.color.rgb = COLOR_ACCENT
            cp.alignment = PP_ALIGN.CENTER
        else:
            cp.font.color.rgb = COLOR_SECONDARY
            cp.alignment = PP_ALIGN.CENTER

# Right Column: 2 Rigor Defense Cards
r_card_w = Inches(4.0)
r_card_left = Inches(8.533)
top_rc1 = Inches(1.68)
h_rc1 = Inches(2.55)

card_rc1 = create_card(slide8, r_card_left, top_rc1, r_card_w, h_rc1)
tf_rc1 = card_rc1.text_frame
tf_rc1.word_wrap = True
tf_rc1.margin_left = tf_rc1.margin_right = Inches(0.22)
tf_rc1.margin_top = Inches(0.16)

p = tf_rc1.paragraphs[0]
p.text = "EW JAMMING & MESH PARTITIONING"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = tf_rc1.add_paragraph()
p.text = "ROBUST GRAPH TOPOLOGY DISRUPTION"
p.font.name = FONT_HEADING
p.font.size = Pt(8.5)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(6)

ew_pts = [
    "Softmax Attention Masking: Jammed links set A_ij = 0; masked attention (-inf) completely drops jammed nodes from denominator.",
    "Partitioned Sub-Swarms: If severe RF jamming splits the mesh into disconnected islands, each sub-swarm optimizes visible targets locally.",
    "Zero Crash Guarantee: System degrades gracefully into localized multi-agent clusters rather than failing catastrophically."
]
for pt in ew_pts:
    p = tf_rc1.add_paragraph()
    p.text = "•  " + pt
    p.font.name = FONT_BODY
    p.font.size = Pt(8.5)
    p.font.color.rgb = COLOR_SECONDARY
    p.space_after = Pt(2)

top_rc2 = top_rc1 + h_rc1 + Inches(0.18)
h_rc2 = Inches(2.57)

card_rc2 = create_card(slide8, r_card_left, top_rc2, r_card_w, h_rc2)
tf_rc2 = card_rc2.text_frame
tf_rc2.word_wrap = True
tf_rc2.margin_left = tf_rc2.margin_right = Inches(0.22)
tf_rc2.margin_top = Inches(0.16)

p = tf_rc2.paragraphs[0]
p.text = "ARBITRARY SWARM SCALABILITY"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = COLOR_PRIMARY

p = tf_rc2.add_paragraph()
p.text = "GENERALIZATION & HEADLESS TESTING"
p.font.name = FONT_HEADING
p.font.size = Pt(8.5)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT
p.space_after = Pt(6)

scale_pts = [
    "Dynamic Swarm Scaling: Evaluates N=3, 4, 8, or 16 interceptors against arbitrary target counts M with zero code changes.",
    "CLI Testing Command:\npy run_simulation.py --drones 6 --targets 4",
    "Headless Benchmark Mode:\npy run_simulation.py --no-gui --save-plot",
    "Deterministic Performance: Sub-2.5 ms execution holds even under high agent density scenarios."
]
for pt in scale_pts:
    p = tf_rc2.add_paragraph()
    p.text = "•  " + pt
    p.font.name = FONT_BODY
    p.font.size = Pt(8.5)
    p.font.color.rgb = COLOR_SECONDARY
    p.space_after = Pt(2)


# ==============================================================================
# SLIDE 9: ARCHITECTURAL SYNTHESIS & STRATEGIC IMPACT
# ==============================================================================
slide9 = prs.slides.add_slide(blank_layout)
apply_light_background(slide9)
add_header(slide9, "Architectural Synthesis: 5 Core Technical Pillars",
           category="STRATEGIC IMPACT // EXECUTIVE SUMMARY", badge="MISSION COMPLETE")

pillars = [
    {
        "num": "01",
        "tag": "DECENTRALIZED AUTONOMY",
        "title": "Mesh Consensus",
        "detail": "Eliminated vulnerable centralized mother craft and C2 ground servers. Swarm achieves robust peer-to-peer coordination across an ad-hoc 12 km RF mesh."
    },
    {
        "num": "02",
        "tag": "GRAPH ATTENTION",
        "title": "Decomposed GAT",
        "detail": "Decomposed attention projection reduces compute from O(|E|) to O(|V|). Learned attention weights dynamically model inter-agent trajectory intentions."
    },
    {
        "num": "03",
        "tag": "ZERO-TRAINING CONSENSUS",
        "title": "Mutual Inhibition",
        "detail": "Analytical inhibitory dynamics eliminate non-convergent Multi-Agent PPO training fragility, guaranteeing deterministic 0.0% overkill allocation in 1.10 ms."
    },
    {
        "num": "04",
        "tag": "SIZE INVARIANCE",
        "title": "Permutation Equivariance",
        "detail": "Neighborhood aggregation operates over local peer sets N_i, enabling instant deployment across 3, 4, 8, or 20 agents without retraining or shape errors."
    },
    {
        "num": "05",
        "tag": "DYNAMIC FAULT TOLERANCE",
        "title": "50 ms Attrition Failover",
        "detail": "Mid-air casualty of lead agent a_0 triggers immediate autonomous role reallocation in a single 50 ms flight cycle, maintaining 100% target coverage."
    }
]

pil_w = Inches(2.18)
pil_h = Inches(4.25)
pil_gap = Inches(0.208)
pil_y = Inches(1.68)

for idx, pil in enumerate(pillars):
    pil_left = Inches(0.8) + idx * (pil_w + pil_gap)
    pcard = create_card(slide9, pil_left, pil_y, pil_w, pil_h, bg_color=COLOR_CARD)
    
    # Top Accent on Pillar Card
    ptop = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, pil_left, pil_y, pil_w, Inches(0.08))
    ptop.fill.solid()
    ptop.fill.fore_color.rgb = COLOR_ACCENT
    ptop.line.color.rgb = COLOR_ACCENT

    ptf = pcard.text_frame
    ptf.word_wrap = True
    ptf.margin_left = ptf.margin_right = Inches(0.16)
    ptf.margin_top = Inches(0.14)

    p0 = ptf.paragraphs[0]
    p0.text = pil["num"]
    p0.font.name = FONT_HEADING
    p0.font.size = Pt(28)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_ACCENT

    p1 = ptf.add_paragraph()
    p1.text = pil["tag"]
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(8)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_ACCENT_MUTED
    p1.space_after = Pt(4)

    p2 = ptf.add_paragraph()
    p2.text = pil["title"]
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_PRIMARY
    p2.space_after = Pt(8)

    p3 = ptf.add_paragraph()
    p3.text = pil["detail"]
    p3.font.name = FONT_BODY
    p3.font.size = Pt(8.8)
    p3.font.color.rgb = COLOR_SECONDARY

# Bottom Executive Synthesis Banner
bot_banner = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.08), Inches(11.733), Inches(0.92))
bot_banner.fill.solid()
bot_banner.fill.fore_color.rgb = COLOR_PRIMARY
bot_banner.line.color.rgb = COLOR_ACCENT
bot_banner.line.width = Pt(1.5)

b_tf = bot_banner.text_frame
b_tf.word_wrap = True
b_tf.margin_left = Inches(0.28)
b_tf.margin_top = Inches(0.12)

bp_top = b_tf.paragraphs[0]
bp_top.text = "EXECUTIVE CONCLUSION: MISSION-READY EDGE SWARM AUTONOMY"
bp_top.font.name = FONT_HEADING
bp_top.font.size = Pt(10.5)
bp_top.font.bold = True
bp_top.font.color.rgb = COLOR_ACCENT

bp_bot = b_tf.add_paragraph()
bp_bot.text = "The integrated GAT + Mutual Inhibition guidance architecture replaces fragile centralized links with robust ad-hoc autonomy. Fully verified via kinematic simulation: 100% intercept success, 0.0% overkill, and sub-2 ms edge decision latency."
bp_bot.font.name = FONT_BODY
bp_bot.font.size = Pt(9)
bp_bot.font.color.rgb = COLOR_WHITE


# ==============================================================================
# SAVE PRESENTATION FILES
# ==============================================================================
output_tech = "Distributed_Swarm_Guidance_Technical.pptx"
output_pres = "Distributed_Swarm_Guidance_Presentation.pptx"

saved_files = []
for fname in [output_tech, output_pres]:
    try:
        prs.save(fname)
        saved_files.append(fname)
        print(f"[OK] Successfully saved: {fname}")
    except PermissionError:
        print(f"[LOCKED] Could not write to {fname} (file open in PowerPoint).")
    except Exception as e:
        print(f"[ERROR] Failed to save {fname}: {e}")

if not saved_files:
    fallback = "Distributed_Swarm_Guidance_Technical_v2.pptx"
    prs.save(fallback)
    print(f"[SAVED] Saved to fallback: {fallback}")

print("\nPresentation generation complete!")
