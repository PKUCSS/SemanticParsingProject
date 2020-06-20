
SemEval 2014 Task 8: Broad-Coverage Semantic Dependency Parsing: Training Data

Version 1.2; January 13, 2014


Overview
========

This directory contains three files, each providing semantic dependency graphs
for a training set of 34003 sentences (745543 tokens), taken from the first 20
sections of the PTB WSJ Corpus.  The files instantiate the tab-separated format
for the description of general directed graphs, as documented on-line:

  http://sdp.delph-in.net/2014/data.html

The three annotation formats have been aligned sentence- and token-wise, i.e.
they annotate the exact same text; however, there are differences in LEMMA and
POS fields.

File names are comprised of an acronym denoting the specific annotation type,
and the common suffix ‘.sdp’ (which we will use on all data for this task).


Recommended Split
=================

If for system development you want to designate part of the training material
as a development set, we strongly recommend using Section 20 for this purpose.


Known Errors
============

None, for the time being.


Contact
=======

For questions or comments, please do not hesitate to email the task organizers
at: ‘sdp-organizers.emmtee.net’.

Dan Flickinger
Jan Hajič
Marco Kuhlmann
Yusuke Miyao
Stephan Oepen
Yi Zhang
Daniel Zeman
