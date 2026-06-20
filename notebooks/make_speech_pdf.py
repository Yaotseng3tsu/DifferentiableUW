"""生成 ~10 分钟英文演讲稿 PDF（带翻页标记 + Q&A），保存到 docs/。

运行：.venv\\Scripts\\python.exe notebooks\\make_speech_pdf.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "presentation_script_0619.pdf")

TITLE = ("Differentiable Simulation of Underwater Robot Dynamics "
         "for Parameter and Flow-Disturbance Identification")

# (slide_no, slide_title, time_cue, [paragraphs])
SLIDES = [
    (1, "Title", "~25s", [
        "Good afternoon, everyone. My name is Zeng. The title of my talk is "
        "\"Differentiable Simulation of Underwater Robot Dynamics for Parameter "
        "and Flow-Disturbance Identification.\"",
        "In one sentence: I am building an underwater-robot simulator whose physics "
        "can be differentiated, so that instead of only predicting motion, we can run "
        "it backwards to recover unknown quantities, such as the robot's hydrodynamic "
        "properties and the water currents acting on it."]),
    (2, "Background", "~60s", [
        "Let me start with why this matters. Real underwater work almost never happens "
        "in still water; there are always currents disturbing the robot.",
        "Underwater robotics faces two bottlenecks. First, data scarcity: collecting "
        "real underwater data is hard, expensive, and risky, with real collision costs "
        "during harbor tests. Second, the sim-to-real gap: today's simulators get the "
        "physics wrong, especially hydrodynamic forces and sensor noise, and there is "
        "often a disconnect between the control software and the real hardware propulsion. "
        "So we want a simulator whose physics we can actually trust."]),
    (3, "Previous Research (1/2)", "~50s", [
        "So what do we have today? Two representative underwater simulators are HoloOcean "
        "and OceanSim. They are impressive; OceanSim in particular gives photorealistic "
        "rendering.",
        "But look at the physics: both model only thrust and buoyancy, with minimal fluid "
        "mechanics, and crucially, neither is differentiable. They are visually rich but "
        "physically hollow. The missing link is a simulator that combines good, physics-based "
        "fluid-robot interaction with differentiability."]),
    (4, "Previous Research (2/2): NVIDIA Warp", "~75s", [
        "Our key enabling tool is NVIDIA Warp. Warp is a Python framework that compiles your "
        "code into GPU kernels, so it runs fast and in parallel. More importantly, it has "
        "built-in automatic differentiation: it computes exact gradients through the whole "
        "physics computation, using what is called a tape backward pass.",
        "Why does differentiability matter? Because it turns the simulator into an inverse "
        "tool: instead of only going from parameters to motion, we can go from observed "
        "motion back to the unknown quantities.",
        "This is not free. There are real challenges we must solve. First, identifiability "
        "and experiment design: not every parameter is observable from every maneuver. "
        "Second, ill-conditioned identification: hydrodynamic parameters span several orders "
        "of magnitude. And third, disturbance-field inversion: recovering a current that "
        "varies in space and time is fundamentally ill-posed."]),
    (5, "Objective", "~20s", [
        "That leads to my objective, stated simply: to develop a differentiable underwater "
        "dynamics simulator that explicitly accounts for fluid disturbances."]),
    (6, "Objective: Goals and Benefits", "~70s", [
        "Let me make that concrete, with four goals. One: build and validate a differentiable "
        "Fossen dynamics model in Warp; Fossen's equations are the standard model for marine "
        "vehicles. Two: identify specific hydrodynamic parameters from trajectories, namely "
        "the added-mass terms and the linear and quadratic damping coefficients. Three: "
        "estimate the fluid disturbance, by recovering the resultant current force with the "
        "differentiable inverse solver, and then working toward a full flow field. And four: "
        "validate everything on real experimental data.",
        "If we succeed, the benefits are concrete too: we can calibrate a robot's dynamics "
        "directly from how it moves, which is data-efficient, and we obtain physics-based "
        "gradients that can drive reinforcement learning and disturbance-aware planning."]),
    (7, "Proposed Method (1/3): Differentiable Fossen Model", "~75s", [
        "Now the method, in three steps, in the order we actually build it. Step one is the "
        "differentiable dynamics model. This is the Fossen equation in the body frame: mass "
        "times acceleration, plus Coriolis, plus damping, plus restoring forces, equals the "
        "control input plus the environmental disturbance.",
        "The mass term includes added mass, which is the extra inertia from the water the "
        "robot drags along, and the damping is both linear and quadratic in velocity. We "
        "implement all of this as differentiable GPU kernels, so we get analytic gradients "
        "automatically. We follow a reduced-dimensionality strategy: first a 3-DOF planar "
        "model, to validate the integration and the gradient pipeline, then we extend to full "
        "6-DOF. The problems we focus on here are gradient correctness, parameter "
        "identifiability, and ill-conditioned optimization."]),
    (8, "Proposed Method (2/3): Disturbance Modeling", "~45s", [
        "Step two is how we model the disturbance, and we go from simple to hard. We start by "
        "simplifying the environmental current into a single resultant force acting at the "
        "robot's center of mass. That is tractable, and as I will show, we can recover it "
        "exactly. The harder, open challenge, and one of our main research directions, is to "
        "extend this to a current that varies in space and time: a full flow field."]),
    (9, "Proposed Method (3/3): Validation with Real Data", "~50s", [
        "Step three closes the loop: validation with real experimental data. We feed measured "
        "trajectories and onboard IMU data into the differentiable simulator, and we check two "
        "things. First, that the simulated dynamics are physically reasonable. Second, that "
        "the parameters and forces our pipeline recovers actually agree with the real data, "
        "and with values computed the classical way. In other words, the differentiable method "
        "should not just run; its answers have to match reality."]),
    (10, "Current Progress", "~70s", [
        "So where are we now? We have built the differentiable 3-DOF Fossen model in Warp, and "
        "we have verified its gradients against finite differences. On synthetic data, our "
        "parameter identification recovers nine hydrodynamic parameters with about 0.6 percent "
        "mean error. Our inverse disturbance solver recovers a constant current exactly; for a "
        "time-varying current it recovers the main part, and we identified a limit: the current "
        "near the end of the trajectory is poorly observed, because there is no later motion to "
        "constrain it. The whole pipeline is robust to measurement noise and to the initial "
        "guess.",
        "I should be clear: these results are on synthetic data. They validate that the "
        "pipeline is correct; validating on real data is the next step."]),
    (11, "Future Plan", "~55s", [
        "Finally, the plan. Until the mini thesis: finish the basic differentiable dynamics "
        "model in Warp, which is largely done; the disturbance integration and inverse solver, "
        "which are working; and then validation on real experimental data.",
        "Longer term, there are two exciting directions: coupling with differentiable sonar "
        "rendering to build an end-to-end perception-to-action chain; and integrating our "
        "differentiable dynamics into NVIDIA Isaac Sim, so we can eventually get high-fidelity "
        "rendering on top of physics we can trust. Thank you. I am happy to take questions."]),
]

QA = [
    ("Why use NVIDIA Warp instead of PyTorch, JAX, or writing CUDA directly?",
     "Warp is designed for exactly this kind of work: you write simulation kernels in Python, "
     "they compile to fast GPU code, and you get automatic differentiation through them. "
     "Compared to raw CUDA it is far less code and we get gradients for free. Compared to "
     "PyTorch or JAX, Warp is built around per-element simulation kernels and spatial data, "
     "which fits physics simulation more naturally. It also lives in the NVIDIA ecosystem, "
     "which matters for our long-term plan to connect with Isaac Sim."),
    ("You simplify the current to a single force. Isn't that too unrealistic?",
     "It is a deliberate first step. A single resultant force at the center of mass is the "
     "simplest model that still captures the net effect of a current, and we can recover it "
     "exactly, which validates the inverse machinery. Real currents vary in space and time, "
     "so extending to a full flow field is a main research direction. We have started, and we "
     "found it is an ill-posed problem that needs regularization."),
    ("How do you know the gradients from the simulator are actually correct?",
     "We check them against finite differences. For every parameter we compare the analytic "
     "gradient from the tape backward pass to a numerical estimate from perturbing that "
     "parameter. They agree for all eleven parameters, which is our gate before trusting any "
     "identification result."),
    ("Your 0.6 percent error is on synthetic data. Why believe it works on real data?",
     "Synthetic results only prove the pipeline is self-consistent. That is why we test "
     "robustness: we add centimeter-level measurement noise, like a real camera-marker "
     "tracker, and we vary the initial guess. The error grows gracefully and stays bounded. "
     "Real-data validation is the explicit next step, and that is what the upcoming data "
     "collection is for."),
    ("Are all the parameters actually identifiable?",
     "No, and that is an important finding. In planar motion, the yaw inertia and the added "
     "yaw inertia only ever appear as a sum, so they cannot be separated from trajectory data "
     "alone. We detect this directly, because their gradients are identical. Our strategy is "
     "to measure what we can measure, like mass and inertia, and identify only the "
     "hard-to-measure terms, like added mass and damping."),
    ("Why does identification depend on the maneuver you choose?",
     "A parameter can only be identified if the motion actually excites it. For example, sway "
     "damping barely affects the trajectory unless the robot moves sideways, so its gradient "
     "is tiny and it is hard to recover. This is why experiment design matters, and it "
     "connects directly to the towing and free-running experiments we plan: each maneuver is "
     "chosen to excite specific parameters."),
    ("How does this compare to classical system identification, like least squares?",
     "For the parts of the Fossen model that are linear in the parameters, classical least "
     "squares works well and is a good baseline. The advantage of the differentiable approach "
     "is generality: it handles the nonlinear couplings, the disturbance-force and flow-field "
     "inversion, and it plugs into the same gradient-based framework we would use later for "
     "control, learning, and even differentiable rendering. So one framework covers all of "
     "these uniformly."),
    ("What is the tail observability limit you mentioned?",
     "When we recover a time-varying current, the current at a given instant only shows up in "
     "the trajectory that comes after it. Near the end of the recording there is almost no "
     "future motion to constrain it, so that part is poorly recovered. It is the time-domain "
     "version of the observability issue, and it suggests we may need longer recordings or "
     "smoothing at the boundaries."),
]

# ---- 样式 -----------------------------------------------------------------
styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=11, leading=16,
                      alignment=TA_JUSTIFY, spaceAfter=8)
title_s = ParagraphStyle("title", parent=styles["Title"], fontSize=16, leading=20)
sub_s = ParagraphStyle("sub", parent=styles["Normal"], fontSize=10, leading=14,
                       textColor=HexColor("#555555"), spaceAfter=14)
slide_h = ParagraphStyle("slideh", parent=styles["Heading2"], fontSize=12.5,
                         textColor=HexColor("#1a4f8a"), spaceBefore=10, spaceAfter=4)
cue_s = ParagraphStyle("cue", parent=styles["Normal"], fontSize=10,
                       textColor=HexColor("#b03030"), spaceAfter=6)
qa_q = ParagraphStyle("qaq", parent=styles["Normal"], fontSize=11, leading=15,
                      textColor=HexColor("#1a4f8a"), spaceBefore=10, spaceAfter=2)
qa_a = ParagraphStyle("qaa", parent=body)

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                        leftMargin=2.2 * cm, rightMargin=2.2 * cm)
flow = []
flow.append(Paragraph("Presentation Script (about 10 minutes)", title_s))
flow.append(Paragraph(TITLE + " &mdash; M1, Group J (2026-06-19)", sub_s))
flow.append(Paragraph("Note: [ CLICK ] marks where to advance the slide. Time cues are "
                      "approximate; total is about 9 to 10 minutes.", sub_s))

for no, stitle, cue, paras in SLIDES:
    flow.append(Paragraph(f"[ CLICK ]  Slide {no}: {stitle}", slide_h))
    flow.append(Paragraph(f"(approx {cue})", cue_s))
    for p in paras:
        flow.append(Paragraph(p, body))

flow.append(PageBreak())
flow.append(Paragraph("Anticipated Questions and Answers", title_s))
flow.append(Spacer(1, 6))
for i, (q, a) in enumerate(QA, 1):
    flow.append(Paragraph(f"Q{i}. {q}", qa_q))
    flow.append(Paragraph(f"A. {a}", qa_a))

doc.build(flow)
print("saved:", OUT)
