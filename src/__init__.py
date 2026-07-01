"""Exact interval denial-constraint discovery prototype."""

from evidence import (
    EvidenceCatalog,
    EvidenceSupportSidecar,
    TDC4_CONTEXT_ROW_THRESHOLD,
    build_evidence_catalog_contexts,
    build_evidence_catalog_contexts_naive,
    build_evidence_catalog_explicit,
    build_evidence_catalog_tdc4_auto,
    build_evidence_catalog_tdc4_contexts,
    build_evidence_catalog_tdc4_default_gt_contexts,
    build_evidence_catalog_tdc4_enum,
)
from mining import IntervalDCResult, mine_global_dcs, mine_interval_dc_regions
from predicates import Predicate, build_predicate_space
from rulestore import RuleStoreIndex, RuleStoreResult, mine_interval_dc_rulestore, query_rule_store
from upper_layer import UpperLayerConfig, UpperLayerResult, run_upper_layer, run_upper_layer_from_rulestore, write_upper_layer_outputs

SpanDCResult = IntervalDCResult

__all__ = [
    "EvidenceCatalog",
    "EvidenceSupportSidecar",
    "IntervalDCResult",
    "SpanDCResult",
    "Predicate",
    "RuleStoreResult",
    "RuleStoreIndex",
    "TDC4_CONTEXT_ROW_THRESHOLD",
    "UpperLayerConfig",
    "UpperLayerResult",
    "build_evidence_catalog_contexts",
    "build_evidence_catalog_contexts_naive",
    "build_evidence_catalog_explicit",
    "build_evidence_catalog_tdc4_auto",
    "build_evidence_catalog_tdc4_contexts",
    "build_evidence_catalog_tdc4_default_gt_contexts",
    "build_evidence_catalog_tdc4_enum",
    "build_predicate_space",
    "mine_global_dcs",
    "mine_interval_dc_regions",
    "mine_interval_dc_rulestore",
    "query_rule_store",
    "run_upper_layer",
    "run_upper_layer_from_rulestore",
    "write_upper_layer_outputs",
]
