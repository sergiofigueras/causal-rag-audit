PYTHON ?= python3
PYTHONPATH_VALUE := src:.

.PHONY: check compile test validate-example demo

compile:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m compileall -q src examples tests

test:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m unittest discover -s tests -v

validate-example:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m causal_rag_audit validate examples/audit_cases.json

demo:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m causal_rag_audit run \
		--dataset examples/audit_cases.json \
		--target examples.demo_rag:grounded_rag \
		--output artifacts/demo \
		--minimum strict_causal_evidence_score=1.0

check: compile test validate-example demo
