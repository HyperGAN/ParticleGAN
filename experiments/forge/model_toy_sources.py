"""Frozen non-code resources read by public model-toy provider discovery.

AE discovery reads both task cards, including when the requested case is the
AE-DDGAN composition. These inputs must travel with the queued source, rather
than resolving against a mutable checkout or failing after reservation.
"""


def model_toy_source_paths(task):
    if task.get("adapter") != "public_api_model_toy":
        return ()
    provider = task.get("execution", {}).get("host_definition", {}).get("provider", "")
    if provider in ("benchmarks.toy_audit.api_model_ae", "benchmarks.toy_audit.api_model_ae_ddgan"):
        return ("configs/forge/tasks/model_ae_reconstruction.json",
                "configs/forge/tasks/model_ae_gan_multimodal.json")
    return ()
