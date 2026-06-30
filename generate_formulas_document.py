#!/usr/bin/env python3
"""Generate Word document mapping ICLR 2024 formulas to main_with_vanilla.py."""

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def add_formula_block(doc, number, title, formula, code_ref, notes=None):
    p = doc.add_paragraph()
    run = p.add_run(f"Equation {number}: {title}")
    run.bold = True
    run.font.size = Pt(11)

    f = doc.add_paragraph()
    f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = f.add_run(formula)
    fr.italic = True
    fr.font.size = Pt(12)

    c = doc.add_paragraph()
    cr = c.add_run(f"Code: {code_ref}")
    cr.font.size = Pt(10)

    if notes:
        n = doc.add_paragraph()
        nr = n.add_run(f"Note: {notes}")
        nr.font.size = Pt(10)
        nr.italic = True

    doc.add_paragraph()


def build_document():
    doc = Document()

    title = doc.add_heading(
        "Mathematical Formulas in main_with_vanilla.py", 0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph(
        "Mapping to: Sun & Zheng (ICLR 2024) — "
        "\"Belief-Enriched Pessimistic Q-Learning Against Adversarial State Perturbations\"\n"
        "Sequential order follows execution flow in main_with_vanilla.py"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # --- Section 1: Background ---
    add_heading(doc, "1. Background (Paper Section 2)", 1)

    add_formula_block(
        doc, "1.1", "MDP Definition",
        "M = ⟨S, A, P, R, γ⟩",
        "GridWorld.__init__, Q_Agent.__init__",
        "State space S (5×5 grid), action space A = {UP, DOWN, LEFT, RIGHT}, "
        "transition P(s′|s,a), reward R(s,a), discount γ.",
    )

    add_formula_block(
        doc, "1.2", "Standard Action-Value Bellman Equation (Definition 1)",
        "Q^π(s,a) = R(s,a) + γ Σ_{s′∈S} P(s′|s,a) [ Σ_{a′∈A} π(a′|s′) Q^π(s′,a′) ]",
        "GridWorld.step, GridWorld.take_action",
        "Environment transition and reward under policy π.",
    )

    add_formula_block(
        doc, "1.3", "Bellman Optimal Operator (Equation 2)",
        "(T* Q)(s,a) = R(s,a) + γ Σ_{s′∈S} P(s′|s,a) max_{a′∈A} Q(s′,a′)",
        "Q_Agent.learn, Q_Agent.learn_with_greedy_target",
        "Vanilla Q-learning target; fixed point Q* satisfies T*Q* = Q*.",
    )

    add_formula_block(
        doc, "1.4", "State Perturbation Constraint (Section 2.2)",
        "s̃_t ∈ B_ε(s_t)",
        "Q_Agent.get_ball, apply_perturbation",
        "Attacker perturbs observation to s̃ within ε-ball B_ε(s). "
        "In this code B_ε(s) is the L1 neighborhood: s plus adjacent grid cells.",
    )

    add_formula_block(
        doc, "1.5", "Attacker MDP Reward (Section 2.2)",
        "R̃(s, s̃) = − Σ_{a∈A} π(a|s̃) R(s,a)",
        "GridWorld_RL_Attack.step",
        "Attacker reward is negative of defender reward; minimizes agent cumulative return.",
    )

    add_formula_block(
        doc, "1.6", "Attacker Transition (Section 2.2)",
        "P̃(s′|s, s̃) = Σ_a π(a|s̃) P(s′|s,a)",
        "GridWorld_RL_Attack.step",
        "True state evolves via defender action at perturbed observation s̃.",
    )

    # --- Section 2: Pessimistic Q-Learning ---
    add_heading(doc, "2. Pessimistic Q-Learning (Paper Section 3.2, Algorithm 1)", 1)

    add_formula_block(
        doc, "2.1", "Stackelberg Agent Policy — Maximin (Algorithm 1, Line 5)",
        "π(s̃) = argmax_{a∈A} min_{s̄∈B_ε(s̃)} Q(s̄, a)",
        "Q_Agent.get_policy, Q_Agent.get_policy_with_belief, Q_Agent.pareto_update",
        "Agent picks action maximizing worst-case Q over possible true states in ε-ball.",
    )

    add_formula_block(
        doc, "2.2", "Attacker Best Response (Algorithm 1, Line 6)",
        "ω_π(s) = argmin_{s̃∈B_ε(s)} Q(s, π(s̃))",
        "Q_Agent.get_att_state, play_att_with_belief (training), GridWorld_RL_Attack.step",
        "Worst-case perturbation minimizing Q under current policy π.",
    )

    add_formula_block(
        doc, "2.3", "ε-Greedy Behavior Policy (Algorithm 1, Lines 8–9)",
        "a = π(s̃) with probability 1−ε′;  otherwise a ∈ A uniform random",
        "Q_Agent.choose_action, Q_Agent.choose_action_att, play, play_att_with_belief",
        "Exploration during training; ε′ annealed via linear_schedule_exp.",
    )

    add_formula_block(
        doc, "2.4", "Pessimistic Q-Learning Update (Algorithm 1, Line 11)",
        "Q(s,a) ← Q(s,a) + α [ R(s,a) + γ Q(s′, π(ω_π(s′))) − Q(s,a) ]",
        "Q_Agent.learn_with_att_with_belief",
        "Tabular pessimistic backup using joint policy π∘ω_π at next state.",
    )

    add_formula_block(
        doc, "2.5", "Pessimistic Bellman Target with Belief (Section 3.5, Algorithm 4 Line 13)",
        "y_i = R_i + γ max_{a′∈A} min_{m∈M_i} Q′(m, a′)",
        "Q_Agent.find_maxmin_with_belief",
        "BP-DQN target; M_i is belief set at transition i.",
    )

    add_formula_block(
        doc, "2.6", "Belief-Enriched CAR Operator (Implementation of Eq. 2.5 in tabular form)",
        "(T_{car} Q)(s,a) = R(s,a) + γ · max_{a′} min_{s′_ν ∈ B_ε(s′) ∩ M_t} Q(s′, a′)",
        "Q_Agent.find_maxmin_with_belief, Q_Agent.learn_with_att_with_belief",
        "Corresponds to the pessimistic contraction operator used in Q-updates. "
        "Code intersects belief M_t with B_ε(s′) via states_intersection.",
    )

    add_formula_block(
        doc, "2.7", "Bellman Operator Under Fixed Joint Policy (Equation 1)",
        "(T^{π∘ω} Q)(s,a) = R(s,a) + γ Σ_{s′∈S} P(s′|s,a) Q(s′, π(ω(s′)))",
        "Q_Agent.learn_with_att_with_belief (expected form); Algorithm 3 Line 7",
        "Value iteration form; tabular code uses sample transitions.",
    )

    # --- Section 3: Belief ---
    add_heading(doc, "3. Belief State Update (Paper Section 3.3, Algorithm 2)", 1)

    add_formula_block(
        doc, "3.1", "Initial Belief",
        "M_0 = B_ε(s̃_0)",
        "play_att_with_belief, GridWorld_RL_Attack.reset",
        "Initially all states in ε-ball around first perturbed observation.",
    )

    add_formula_block(
        doc, "3.2", "Reachable Next States (Algorithm 2, Lines 1–8)",
        "M′_t = { s′ ∈ S : ∃ s ∈ M_t, P(s′|s, a_t) > 0 }",
        "play_att_with_belief (belief propagation loop), states_intersection",
        "All next states reachable from belief under action a_t.",
    )

    add_formula_block(
        doc, "3.3", "Belief Intersection Update (Algorithm 2, Line 9)",
        "M_{t+1} = M′_t ∩ B_ε(s̃_{t+1})",
        "states_intersection, play_att_with_belief, play_offpolicy_greedy_target (training)",
        "Narrows belief using new perturbed observation s̃_{t+1}.",
    )

    add_formula_block(
        doc, "3.4", "Belief-Restricted Maximin Policy (Algorithm 5, Line 7)",
        "a_t = argmax_{a∈A} min_{m∈M_t} Q_r(m, a)",
        "Q_Agent.get_policy_with_belief, Q_Agent.choose_action_att",
        "Testing policy with belief M_t. Code uses mean over M_t∩B_ε(s̃) as a "
        "tabular approximation when |M_t∩B_ε|>1.",
    )

    # --- Section 4: Vanilla baseline ---
    add_heading(doc, "4. Vanilla Q-Learning Baseline", 1)

    add_formula_block(
        doc, "4.1", "Vanilla Q-Learning Update",
        "Q(s,a) ← (1−α)Q(s,a) + α [ R + γ max_{a′} Q(s′,a′) ]",
        "Q_Agent.learn, play",
        "Standard temporal-difference backup; no robustness to perturbations.",
    )

    add_formula_block(
        doc, "4.2", "Greedy Test Policy (train/test mismatch criticized in paper)",
        "a = argmax_{a∈A} Q(s̃, a)",
        "play_vanilla_test, play_pessimistic_clean, play_offpolicy_greedy_target (test)",
        "Paper notes using true s at train but s̃ at test causes brittleness.",
    )

    # --- Section 5: Off-policy extension ---
    add_heading(doc, "5. Off-Policy Greedy Target Extension (This Codebase)", 1)

    add_formula_block(
        doc, "5.1", "Off-Policy Decomposition",
        "Behavior: π_behavior = maximin with belief (Eq. 2.1, 3.4);  "
        "Target: π_target = greedy (Eq. 4.1)",
        "play_offpolicy_greedy_target",
        "Extension not in ICLR paper: pessimistic exploration, greedy bootstrap.",
    )

    add_formula_block(
        doc, "5.2", "Off-Policy Greedy Target Update",
        "Q(s,a) ← (1−α)Q(s,a) + α [ R + γ max_{a′} Q(s′,a′) ]",
        "Q_Agent.learn_with_greedy_target",
        "Same as Eq. 4.1 but updates q_table_withatt during robust training.",
    )

    # --- Section 6: Environment ---
    add_heading(doc, "6. Environment and Attack Simulation", 1)

    add_formula_block(
        doc, "6.1", "Deterministic Grid Transition",
        "s′ = T(s, a)  (border: s′ = s)",
        "GridWorld.step, GridWorld.take_action, apply_perturbation",
        "Deterministic P(s′|s,a) ∈ {0,1} on 5×5 grid.",
    )

    add_formula_block(
        doc, "6.2", "Terminal Condition",
        "terminated = 1[ s ∈ {s_bomb, s_gold} ]",
        "GridWorld.step, GridWorld.check_state",
        "Episode ends at bomb (R=−10) or gold (R=+10).",
    )

    add_formula_block(
        doc, "6.3", "Learning Rate / Exploration Schedule",
        "ε_t = max(ε_min, (T−t)·ε_init / T)",
        "linear_schedule_exp",
        "Implementation helper for annealing α and ε′.",
    )

    add_formula_block(
        doc, "6.4", "PA-AD Attacker Training MDP (Section 2.2, Sun et al. 2021)",
        "Train ω via RL: min_ω Σ_t R̃(s_t, ω(s_t))",
        "train_RL_attack_with_belief, GridWorld_RL_Attack",
        "DQN learns attack policy ω against fixed defender.",
    )

    # --- Section 7: Experiment ---
    add_heading(doc, "7. Experiment Pipeline", 1)

    add_formula_block(
        doc, "7.1", "Performance Difference Metrics",
        "Δ_{off,pes} = R̄_{off,clean} − R̄_{pes,clean}",
        "run_single_experiment_with_seed (Phase 4)",
        "Logged to CSV as performance_diff_offpolicy_vs_pessimistic.",
    )

    add_formula_block(
        doc, "7.2", "Inference Speedup",
        "Speedup = t_{pessimistic} / t_{offpolicy}",
        "run_single_experiment_with_seed (Phase 4)",
        "Computed from clean-test inference times in CSV.",
    )

    add_formula_block(
        doc, "7.3", "Stackelberg Equilibrium Condition (Definition 2)",
        "∀s ∈ S, ∀π:  V^{π*∘ω_{π*}}(s) ≥ V^{π∘ω_π}(s)",
        "Conceptual goal of pessimistic training",
        "Pessimistic Q-learning approximates leader policy in Stackelberg game.",
    )

    out = "/Users/adithya/Education/RL/Pessimistic_Q_Learning_Formulas.docx"
    doc.save(out)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    build_document()
