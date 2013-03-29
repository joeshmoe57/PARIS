SHELL = /bin/sh

LATEX = latex
PDFLATEX = pdflatex
BIBTEX = bibtex
DVIPDF = dvipdf
SPELL = aspell
SPFLAGS = -c --mode=tex

TITLE = Paris
ALL: paper

all: $(ALL)

paper: $(TITLE).tex 
	$(SPELL) $(SPFLAGS) $(TITLE).tex
	$(LATEX) $(TITLE).tex
	$(BIBTEX) $(TITLE).aux
	$(LATEX) $(TITLE).tex
	$(LATEX) $(TITLE).tex

pdf: $(TITLE).tex
	$(SPELL) $(SPFLAGS) $(TITLE).tex
	$(PDFLATEX) $(TITLE).tex
	$(BIBTEX) $(TITLE).aux
	$(PDFLATEX) $(TITLE).tex
	$(PDFLATEX) $(TITLE).tex
