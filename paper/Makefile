LATEX ?= pdflatex
MAINFILE ?= main

.PHONY: all clean

all: $(MAINFILE).pdf

$(MAINFILE).pdf: $(MAINFILE).tex references.bib
	$(LATEX) $(MAINFILE).tex
	bibtex $(MAINFILE)
	$(LATEX) $(MAINFILE).tex
	$(LATEX) $(MAINFILE).tex

clean:
	rm -f $(MAINFILE).aux $(MAINFILE).bbl $(MAINFILE).blg $(MAINFILE).log $(MAINFILE).out
