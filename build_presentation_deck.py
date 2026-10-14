"""
Automated PowerPoint Presentation Generator for Distributed Cooperative Swarm Guidance (DCSG).
Generates an 8-slide, 16:9 widescreen presentation deck with embedded figures,
styled dark-theme aesthetics, and attached speaker notes.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Color Palette (Tailwind Slate / Dark Theme)
    BG_DARK = RGBColor(15, 23, 42)       # #0f172a
    CARD_BG = RGBColor(30, 41, 59)       # #1e293b
    CYAN_ACCENT = RGBColor(56, 189, 248) # #38bdf8
    EMERALD_GREEN = RGBColor(34, 197, 94)# #22c55e
    TEXT_LIGHT = RGBColor(241, 245, 249) # #f1f5f9
    TEXT_MUTED = RGBColor(148, 163, 184) # #94a3b8
    GOLD_ACCENT = RGBColor(251, 191, 36) # #fbbf24

    figures_dir = os.path.join(os.path.dirname(__file__), "figures")

    def add_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="MACHINE LEARNING THEORY DEFENSE"):
        # Category Tracker Pill
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = CYAN_ACCENT

        # Slide Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    add_slide_background(s1)

    tbox1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.5))
    tf1 = tbox1.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "AUTONOMOUS MULTI-AGENT SYSTEMS & MACHINE LEARNING THEORY"
    p_badge.font.size = Pt(13)
    p_badge.font.bold = True
    p_badge.font.color.rgb = CYAN_ACCENT

    p_title = tf1.add_paragraph()
    p_title.text = "Distributed Cooperative Swarm Guidance (DCSG)"
    p_title.font.size = Pt(38)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_LIGHT

    p_sub = tf1.add_paragraph()
    p_sub.text = "Dynamic Weapon-Target Allocation via Graph Attention Networks (GAT) & Multi-Agent PPO"
    p_sub.font.size = Pt(20)
    p_sub.font.color.rgb = EMERALD_GREEN

    p_author = tf1.add_paragraph()
    p_author.text = "\nPresenter: Nunna Mokshith | B.Tech Robotics & Artificial Intelligence\nProject Defense for Machine Learning Theory Course"
    p_author.font.size = Pt(14)
    p_author.font.color.rgb = TEXT_MUTED

    s1.notes_slide.notes_text_frame.text = (
        "Opening (1 minute): Good morning Professor. Today I am presenting Distributed Cooperative "
        "Swarm Guidance using Graph Attention Networks and Multi-Agent Reinforcement Learning. "
        "We are addressing the core multi-agent challenge: how a supersonic swarm of autonomous missiles "
        "can coordinate target selection without a central commander, preventing greedy saturation overkill."
    )

    # ==========================================
    # SLIDE 2: THE PROBLEM (GREEDY HOMOGENEOUS COLLAPSE)
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_background(s2)
    add_header(s2, "The Core ML Problem: Swarm Coordination Collapse")

    c1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    c1.fill.solid(); c1.fill.fore_color.rgb = CARD_BG; c1.line.color.rgb = CYAN_ACCENT
    tf_c1 = c1.text_frame; tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "The Operational Problem: Decentralized WTA"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    bullets_c1 = [
        "Scenario: 4 supersonic missiles (Mach 1.5) attacking 3 targets (SAM radar, Command bunker, Fuel depot).",
        "EW Jammed Environment: Central AWACS/Satellite links are disabled; missiles can only communicate via a 1-hop ad-hoc local mesh.",
        "Greedy Homogeneous Collapse: If each agent independently picks argmax Q(s, a), ALL 4 missiles pick Target 0 (Radar: +10 pts).",
        "Result: Severe Overkill [0, 0, 0, 0]. Secondary critical targets survive completely untouched."
    ]
    for b in bullets_c1:
        p = tf_c1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13); p.font.color.rgb = TEXT_LIGHT

    c2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    c2.fill.solid(); c2.fill.fore_color.rgb = CARD_BG; c2.line.color.rgb = GOLD_ACCENT
    tf_c2 = c2.text_frame; tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "Why Classical Optimization Fails"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = GOLD_ACCENT

    bullets_c2 = [
        "Combinatorial WTA is NP-Complete: Finding optimal allocation over dynamic time horizons scales factorially.",
        "Latency Bottleneck: Mixed-Integer Linear Programming (MILP) or Hungarian algorithms take 500ms - 5000ms to solve.",
        "Supersonic Deadline: At 500 m/s, guidance autopilots demand a strict < 10ms execution loop. Classical solvers cause flight instability.",
        "Single Point of Failure: Centralized optimizers collapse if the master node is destroyed.",
        "Our Solution: Edge-deployed Graph Neural Network with < 2.4ms inference."
    ]
    for b in bullets_c2:
        p = tf_c2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13); p.font.color.rgb = TEXT_LIGHT

    s2.notes_slide.notes_text_frame.text = (
        "Slide 2 Script: In multi-agent systems, if agents optimize selfishly, they suffer from "
        "Greedy Homogeneous Collapse. If one target has higher reward, all agents saturate that single target. "
        "Classical combinatorial algorithms like MILP take up to 5 seconds, which is 500 times too slow for a "
        "10ms missile flight computer cycle. This is why a learned, distributed policy is essential."
    )

    # ==========================================
    # SLIDE 3: MATHEMATICAL FORMULATION
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_background(s3)
    add_header(s3, "Mathematical Foundations: Dec-POMDP & Graph Theory")

    box3 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    box3.fill.solid(); box3.fill.fore_color.rgb = CARD_BG; box3.line.color.rgb = CYAN_ACCENT
    tf3 = box3.text_frame; tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "Formal Multi-Agent Reinforcement Learning Formulation"
    p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    marl_points = [
        ("Dec-POMDP Tuple", "Formulated as < S, {A_i}, P, {R_i}, {Omega_i}, {O_i}, gamma >. Each missile i receives partial observation o_i (onboard radar + neighbor radio packets) and selects discrete target choice a_i in {T0, T1, T2}."),
        ("Dynamic Ad-Hoc Topology", "Swarm graph G(t) = (V, E(t)) with Adjacency A(t) and Degree matrix D(t). Graph Laplacian L = D - A."),
        ("Algebraic Connectivity", "The second smallest eigenvalue of L, lambda_2(L) (Fiedler value = 4.0), mathematically proves the ad-hoc mesh remains strictly connected across the entire cruise phase."),
        ("Separation of Concerns", "Tier 2 (ML Policy) performs discrete high-level target allocation; Tier 1 (Pure Proportional Navigation) executes continuous low-level fin steering (clamped to 25G).")
    ]
    for title, desc in marl_points:
        p_t = tf3.add_paragraph()
        p_t.text = "• " + title + ": "
        p_t.font.bold = True; p_t.font.size = Pt(13); p_t.font.color.rgb = EMERALD_GREEN
        p_t.text += desc
        p_t.font.bold = False

    s3.notes_slide.notes_text_frame.text = (
        "Slide 3 Script: We rigorously model the system as a Dec-POMDP. Missiles only have local observations. "
        "We monitor the algebraic connectivity via the Graph Laplacian Fiedler value lambda_2(L) = 4.0, "
        "guaranteeing communication consensus. Furthermore, we decouple high-level ML target allocation from low-level "
        "classical Proportional Navigation to preserve deterministic flight stability."
    )

    # ==========================================
    # SLIDE 4: NEURAL ARCHITECTURE (GAT)
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_background(s4)
    add_header(s4, "Neural Architecture: Graph Attention Network (GAT) Policy")

    # Insert Fig 3 image if exists
    f3_path = os.path.join(figures_dir, "fig3_neural_architecture.png")
    if os.path.exists(f3_path):
        s4.shapes.add_picture(f3_path, Inches(0.8), Inches(1.6), width=Inches(6.2))

    tbox4 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(1.6), Inches(5.2), Inches(5.2))
    tbox4.fill.solid(); tbox4.fill.fore_color.rgb = CARD_BG; tbox4.line.color.rgb = CYAN_ACCENT
    tf4 = tbox4.text_frame; tf4.word_wrap = True

    p = tf4.paragraphs[0]
    p.text = "GAT Message Passing & Invariance"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    gat_bullets = [
        "Dynamic Attention Weights: alpha_ij = Softmax(LeakyReLU(a^T [W h_i || W h_j])). Missiles dynamically weight neighbor communications based on relative distance and velocity vectors.",
        "Latent Intent Sharing: Neighbor embeddings convey trajectory inclination, allowing agents to sense when a teammate has already committed to Target 0.",
        "Overcoming GNN Over-Smoothing: Residual self-connection h_out = [h_self || sum_j alpha_ij h_j] ensures node embeddings do not collapse to graph centroid.",
        "Permutation Invariance: Handled seamlessly if an agent is shot down (N = 4 -> 3) in 0.58ms without network retraining."
    ]
    for b in gat_bullets:
        p = tf4.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT

    s4.notes_slide.notes_text_frame.text = (
        "Slide 4 Script: Why GAT instead of standard MLPs? MLPs require fixed input dimensions and fail if a missile "
        "is destroyed. GAT provides permutation equivariance. By adding a residual self-connection, we prevent the "
        "classic GNN over-smoothing problem. In our benchmarks, dynamic attrition resizing N=4 to 3 runs in just 0.58 ms."
    )

    # ==========================================
    # SLIDE 5: TRAINING PARADIGM & REWARD SHAPING
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_background(s5)
    add_header(s5, "Training Engine: MAPPO with CTDE & Reward Shaping")

    c5_1 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    c5_1.fill.solid(); c5_1.fill.fore_color.rgb = CARD_BG; c5_1.line.color.rgb = CYAN_ACCENT
    tf5_1 = c5_1.text_frame; tf5_1.word_wrap = True
    p = tf5_1.paragraphs[0]
    p.text = "CTDE: Centralized Training Decentralized Execution"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    ctde_bullets = [
        "The Non-Stationarity Dilemma: Independent Q-learning breaks Markov assumptions because neighbor policies change during training.",
        "Centralized Critic V_phi(S): Observes global joint state during offline training simulation to stabilize Generalized Advantage Estimation (GAE).",
        "Decentralized Actor pi_theta(a_i | o_i): Deployed onboard. Relies exclusively on local radar and 1-hop mesh radio packet.",
        "PPO Clipped Objective: Limits policy updates to [1-eps, 1+eps], preventing destructive divergence."
    ]
    for b in ctde_bullets:
        p = tf5_1.add_paragraph(); p.text = "• " + b; p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT

    c5_2 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    c5_2.fill.solid(); c5_2.fill.fore_color.rgb = CARD_BG; c5_2.line.color.rgb = GOLD_ACCENT
    tf5_2 = c5_2.text_frame; tf5_2.word_wrap = True
    p = tf5_2.paragraphs[0]
    p.text = "Multi-Objective Reward Formulation"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = GOLD_ACCENT

    reward_bullets = [
        "Team Objective: R = R_priority + R_coverage - R_overkill + R_distance.",
        "Priority Score: Sum of targeted assets (Radar: +10, Command: +6, Fuel: +6).",
        "Coverage Bonus: +50 for complete coverage of all 3 targets.",
        "Overkill Penalty: -30 per redundant duplicate agent on the same target.",
        "Gradient Impact: If all 4 pick T0, Team Reward = -45.33. When bifurcated [T0, T2, T1, T0], Reward jumps to +60.00."
    ]
    for b in reward_bullets:
        p = tf5_2.add_paragraph(); p.text = "• " + b; p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT

    s5.notes_slide.notes_text_frame.text = (
        "Slide 5 Script: The engine uses Centralized Training with Decentralized Execution. "
        "The centralized critic eliminates non-stationarity during training. The multi-objective reward applies "
        "an asymmetric penalty for duplicate target assignments. The policy gradient forcefully drives probability mass "
        "toward full target dispersion without any human-crafted heuristic rules."
    )

    # ==========================================
    # SLIDE 6: EMPIRICAL RESULTS (BEFORE VS AFTER)
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_background(s6)
    add_header(s6, "Empirical ML Results: Cooperative Target Allocation")

    f_alloc = os.path.join(figures_dir, "minimal_allocation_visual.png")
    if os.path.exists(f_alloc):
        s6.shapes.add_picture(f_alloc, Inches(0.8), Inches(1.6), width=Inches(8.4))

    tbox6 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.5), Inches(1.6), Inches(3.0), Inches(5.2))
    tbox6.fill.solid(); tbox6.fill.fore_color.rgb = CARD_BG; tbox6.line.color.rgb = EMERALD_GREEN
    tf6 = tbox6.text_frame; tf6.word_wrap = True

    p = tf6.paragraphs[0]
    p.text = "Key Takeaways"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = EMERALD_GREEN

    takeaways = [
        "Before Training: [T0, T0, T0, T0] with reward -45.33 (Greedy Overkill).",
        "After Training: [T0, T2, T1, T0] with reward +60.00 (100% Coverage).",
        "Tactical Bifurcation: M0 & M3 saturate Radar, M1 takes Fuel, M2 strikes Command.",
        "Zero Human Rules: Dispersal is discovered entirely via policy gradients."
    ]
    for t in takeaways:
        p = tf6.add_paragraph(); p.text = "• " + t; p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT

    s6.notes_slide.notes_text_frame.text = (
        "Slide 6 Script: Here is our primary empirical ML result. On the left, untrained agents collapse on T0. "
        "On the right, after training, the agents bifurcate cleanly. Missiles 0 and 3 attack the radar, missile 1 attacks "
        "the fuel depot, and missile 2 attacks the command bunker, achieving 100% coverage and zero target neglect."
    )

    # ==========================================
    # SLIDE 7: CLOSED-LOOP GUIDANCE & TELEMETRY
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_background(s7)
    add_header(s7, "Closed-Loop Flight Execution & Telemetry Dashboard")

    f4_path = os.path.join(figures_dir, "fig4_telemetry_dashboard.png")
    if os.path.exists(f4_path):
        s7.shapes.add_picture(f4_path, Inches(0.8), Inches(1.6), width=Inches(7.6))

    tbox7 = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.7), Inches(1.6), Inches(3.8), Inches(5.2))
    tbox7.fill.solid(); tbox7.fill.fore_color.rgb = CARD_BG; tbox7.line.color.rgb = CYAN_ACCENT
    tf7 = tbox7.text_frame; tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "4-Panel Telemetry Validation"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    telem_points = [
        "1. Dynamic Bifurcation: Missiles cruise in formation, then split cleanly to distinct targets.",
        "2. Inter-Agent Separation: Maintained > 295m (strictly above the 200m safety line).",
        "3. Algebraic Connectivity: Fiedler value lambda_2 = 4.0 verifies mesh link integrity.",
        "4. Fin Acceleration: Demanded G-loads stay below 11.4G (well within the 25G airframe limit)."
    ]
    for tp in telem_points:
        p = tf7.add_paragraph(); p.text = "• " + tp; p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT

    s7.notes_slide.notes_text_frame.text = (
        "Slide 7 Script: We validate the ML target outputs inside a high-fidelity 100 Hz non-holonomic physics simulation. "
        "Panel 1 shows physical trajectory bifurcation. Panel 2 shows inter-agent separation never violates the 200m safety bubble. "
        "Panel 3 proves algebraic connectivity lambda_2(L) is preserved. Panel 4 confirms autopilot fin commands stay under 12G."
    )

    # ==========================================
    # SLIDE 8: HERO BENCHMARKS & CONCLUSION
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_background(s8)
    add_header(s8, "Performance Benchmarks & Summary")

    tbox8_1 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tbox8_1.fill.solid(); tbox8_1.fill.fore_color.rgb = CARD_BG; tbox8_1.line.color.rgb = EMERALD_GREEN
    tf8_1 = tbox8_1.text_frame; tf8_1.word_wrap = True
    p = tf8_1.paragraphs[0]
    p.text = "Empirical Benchmark Scorecard"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = EMERALD_GREEN

    metrics = [
        ("Target Coverage", "100.0% (3 / 3 Targets Neutralized)"),
        ("Overkill Redundancy", "0 Unassigned Enemy Assets"),
        ("Terminal Miss Distance", "0.892 meters (Kinetic Direct Hit)"),
        ("GAT Forward Inference", "2.4 ms (Deadline: 10 ms)"),
        ("Dynamic Attrition Adaptation", "0.58 ms (N = 4 -> 3 missiles)"),
        ("Minimum Swarm Separation", "295.1 m (Safety Bubble: 200 m)")
    ]
    for m_label, m_val in metrics:
        p = tf8_1.add_paragraph()
        p.text = "• " + m_label + ": "
        p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = CYAN_ACCENT
        p.text += m_val
        p.font.bold = False

    tbox8_2 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tbox8_2.fill.solid(); tbox8_2.fill.fore_color.rgb = CARD_BG; tbox8_2.line.color.rgb = CYAN_ACCENT
    tf8_2 = tbox8_2.text_frame; tf8_2.word_wrap = True
    p = tf8_2.paragraphs[0]
    p.text = "Project Deliverables & Artifacts"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    deliverables = [
        ("GitHub Repository", "https://github.com/Nunna-Mokshith/Distributed-Cooperative-Swarm-Guidance"),
        ("Interactive Google Colab", "Self-contained .ipynb notebook with pure ML demo and full salvo simulation"),
        ("Key Architectural Takeaway", "Proves that separating high-level ML combinatorial decision-making from low-level classical control provides both real-time speed (<2.4ms) and guaranteed physical safety.")
    ]
    for d_title, d_desc in deliverables:
        p = tf8_2.add_paragraph()
        p.text = "• " + d_title + ": "
        p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = GOLD_ACCENT
        p.text += d_desc
        p.font.bold = False

    s8.notes_slide.notes_text_frame.text = (
        "Slide 8 Script: To conclude, our hybrid architecture delivers sub-meter accuracy (0.892m) "
        "and sub-3 millisecond neural inference, adapting instantly to agent attrition. All code, "
        "including the minimal GAT PyTorch implementation and the Google Colab notebook, is live and public on GitHub. "
        "Thank you Professor, I am now ready for your questions."
    )

    out_path = os.path.join(os.path.dirname(__file__), "Distributed_Swarm_Guidance_Presentation.pptx")
    prs.save(out_path)
    print(f"Presentation deck successfully created at:\n{out_path}")

if __name__ == "__main__":
    create_deck()
