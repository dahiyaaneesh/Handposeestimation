.PHONY: init clean requirements env

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Initialize project (requirements + io/ folder)
init: requirements

## Install Python dependencies
requirements:
ifdef VIRTUAL_ENV
	python3 -m pip install -U pip setuptools wheel
	python3 -m pip install -r requirements.txt
	ipython kernel install --user --name=thesis
	jupyter contrib nbextension install --user
	jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
	jupyter nbextension enable jupyter-black-master/jupyter-black
	jupyter nbextension enable codefolding/main
	jupyter nbextension enable collapsible_headings/main
	jupyter nbextension enable execute_time/ExecuteTime
	jupyter nbextension enable varInspector/main
	jupyter nbextension enable highlight_selected_word/main
	jupyter nbextension enable --py widgetsnbextension
	jupyter nbextension enable autosavetime/main
	jupyter nbextension enable toc2/main
	python3 -m pip install jupyterthemes
	python3 -m pip install --upgrade jupyterthemes
	jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T
	pre-commit install
else
	@echo "Please create your virtual environment and activate it first (make env; source  master_thesis_env/bin/activate)."
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Set up Python environment
env:
ifndef VIRTUAL_ENV
	python3 -m venv master_thesis_env
else
	@echo "You are already in a virtual environment."
endif


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
