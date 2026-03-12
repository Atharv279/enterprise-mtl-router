import os
import torch
import numpy as np
import json

from src.schemas import Complaint, Officer
from src.models.embedding import SemanticVectorizer
from src.models.mtl_network import ComplaintMTLNetwork
from src.routing.optimizer import route_complaints_optimally


def calculate_cost_matrix(complaint_vector, officers):
    """Calculates inverse cosine similarity to generate a cost matrix."""
    matrix = []
    comp_vec = np.array(complaint_vector)

    for officer in officers:
        off_vec = np.array(officer.skill_vector)

        sim = np.dot(comp_vec, off_vec) / (
            np.linalg.norm(comp_vec) * np.linalg.norm(off_vec)
        )

        cost = 1.0 / (sim + 1e-9)
        matrix.append(cost)

    return [matrix]


if __name__ == "__main__":

    print("Initializing Offline Auto-Routing Pipeline...")
    device = torch.device("cpu")

    # -----------------------------
    # Initialize Vectorizer Early
    # -----------------------------
    print("\nLoading local embedding model...")
    vectorizer = SemanticVectorizer(device=device)

    # -----------------------------
    # 1. Mock available officers
    # -----------------------------
    print("\nLoading active officers and vectorizing their profiles...")

    officers = [
        Officer(
            expertise_profile="Network Infrastructure Expert, Server Hardware, Firewalls, Emergency Response",
            skill_vector=vectorizer.encode("Network Infrastructure Expert, Server Hardware, Firewalls, Emergency Response"),
        ),
        Officer(
            expertise_profile="Hardware Maintenance, Routine Updates, Cabling",
            skill_vector=vectorizer.encode("Hardware Maintenance, Routine Updates, Cabling"),
        ),
        Officer(
            expertise_profile="General Admin, Password Resets, Software Inquiries",
            skill_vector=vectorizer.encode("General Admin, Password Resets, Software Inquiries"),
        ),
    ]

    # -----------------------------
    # 2. Incoming complaint
    # -----------------------------
    test_complaint = Complaint(
        raw_text="The server rack in sector 4 is overheating. Smoke is visible. We need immediate assistance."
    )

    print(f"\n[INCOMING COMPLAINT]: {test_complaint.raw_text}")

    # -----------------------------
    # 3. Generate embedding for Complaint
    # -----------------------------
    print("\n[STEP 1] Generating Semantic Embedding for Complaint...")
    test_complaint.semantic_vector = vectorizer.encode(test_complaint.raw_text)

    # -----------------------------
    # 4. Run MTL network
    # -----------------------------
    print("[STEP 2] Executing MTL Network for Priority & ETA...")

    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(project_root, "data", "processed", "mtl_weights.pth")

    mtl_net = ComplaintMTLNetwork().to(device)

    mtl_net.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )

    mtl_net.eval()

    with torch.no_grad():

        tensor_vec = torch.tensor(
            [test_complaint.semantic_vector], dtype=torch.float32
        ).to(device)

        priority_logits, eta_pred = mtl_net(tensor_vec)

        pred_class_idx = torch.argmax(priority_logits, dim=1).item()

        priority_map = {0: "High", 1: "Medium", 2: "Low"}

        test_complaint.predicted_priority = pred_class_idx + 1
        test_complaint.estimated_eta = round(eta_pred.item(), 2)

    print(f" -> Predicted Priority: {priority_map[pred_class_idx]}")
    print(f" -> Estimated Resolution: {test_complaint.estimated_eta} days")

    # -----------------------------
    # 5. Routing optimization
    # -----------------------------
    print("[STEP 3] Running OR-Tools Constraint Optimization...")

    cost_matrix = calculate_cost_matrix(test_complaint.semantic_vector, officers)

    assignments = route_complaints_optimally(
        cost_matrix,
        max_capacity_per_officer=5
    )

    # -----------------------------
    # 6. Final assignment
    # -----------------------------
    assigned_officer_idx = assignments.get(0)

    if assigned_officer_idx is not None:

        test_complaint.assigned_officer = officers[
            assigned_officer_idx
        ].officer_id

        assigned_profile = officers[assigned_officer_idx].expertise_profile

        print(
            f" -> Assigned to Officer: {test_complaint.assigned_officer} ({assigned_profile})"
        )

    print("\n[PIPELINE COMPLETE] Final JSON Payload:")

    # Fixed Pydantic Deprecation Warning
    payload = test_complaint.model_dump(exclude={"semantic_vector"})

    payload["complaint_id"] = str(payload["complaint_id"])
    payload["timestamp"] = payload["timestamp"].isoformat()

    if payload["assigned_officer"]:
        payload["assigned_officer"] = str(payload["assigned_officer"])

    print(json.dumps(payload, indent=2))