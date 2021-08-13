project_name=tf2bert
project_path=github.com/allenwind/${project_name}

.PHONY: clean
clean:
	rm -rf temp

.PHONY: pyclean
pyclean:
	find . -type d -name "__pycache__" | xargs rm -rf
	find . -name "*.pyc" | xargs rm -rf

.PHONY: gitclean
gitclean:
	find . -name "*.pyc" -exec git rm -f "{}" \;

.PHONY: fmt
fmt:
	python3 -m autopep8 --in-place --recursive .

.PHONY: update
update:
	git pull
