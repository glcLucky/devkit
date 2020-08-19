build: cleanup pywheel
test: tox
all: tox cleanup pywheel

style:
	@echo "### running pycodestyle ..."
	@pycodestyle devkit/
	@echo "### running pyflakes ..."
	@pyflakes devkit/
	@echo "### running perl -wc ..."
	@echo "### finished style ..."

tox:
	@echo "### running tox ..."
	@tox

cleanup:
	@echo "### rm -rf ./build ./dist"
	@rm -rf ./build ./dist

pywheel:
	@echo "### packaging python ..."
	@python setup.py bdist_wheel
