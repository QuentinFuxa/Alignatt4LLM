LATEX=pdflatex
MAINFILE=MAIN

default: $(MAINFILE).pdf

$(MAINFILE).pdf: biblio.bib


bibkey-to-year: biblio.bib
	cat $< | remove_blank_lines | sed 's/^@/\n@/;y/\t/ /' | deblockize | remove_blank_lines | pickre --re='year[=" {]+([12][0-9]{3})[^0-9]' | pickre --re='@[a-zA-Z]+{([^,]+),' \
	> $@

%.pdf: %.tex
	$(LATEX) $<
	bibtex $* || ( echo "Bibtex failed" && exit 1 )
	# change the exit 1 to exit 0 in the line above if you want to ignore it
	lim=4; \
	  while [ $$lim -ge 0 ] \
	      && grep 'Rerun to get\|Citation.*undefined' $*.log >/dev/null 2>/dev/null; do \
	    $(LATEX) $< ; \
	    lim=$$(($$lim - 1)) ; \
	  done

.PHONY: ci
ci:
	# make sure git will remember my password
	git config credential.helper store
	# git pull push dance with possibly uncommited local modifications
	if git pull --no-edit; then echo No changes to hide; else git stash; git pull --no-edit; git stash apply; fi; git commit -am "make ci by $(USER)"; git push

clean:
	rm -f $(MAINFILE).log $(MAINFILE).aux $(MAINFILE).bbl $(MAINFILE).blg $(MAINFILE).toc $(MAINFILE).out $(MAINFILE).run.xml $(MAINFILE).bcf
