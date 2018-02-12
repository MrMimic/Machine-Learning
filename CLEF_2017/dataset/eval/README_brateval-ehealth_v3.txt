BRAT-Eval ehealth
--------------

This tool performs pairwise comparison of annotation sets done on the same set of documents.
The annotated sets are required to be in the BRAT stand-off annotation format (http://brat.nlplab.org/standoff.html).
The current version of the tool has been tested on annotations made with Brat v1.2.
The tool only needs the jar file brateval.jar to work, which is included in the distribution file, and no further libraries are required.
The jar file contains the compiled java classes and the java source files.
In the following examples we assume that the jar file is in the directory from which the java program is called, adjust the classpath parameter (-cp) accordingly.

----------------------------------------------------
CHANGES

v2.0 : evaluation of the attributes + modification output (by Louise Deléger @ LIMSI 2014)
ehealth: evaluation of the AnnotatorNotes + modification of Annotations.read class (by Xavier Tannier @ LIMSI 2015)
ehealth V2: fix CompareNotesEHealth; correct command line instructions (reverse evaluation and reference set folders) - LIMSI 2015
ehealth V3: Additional fixes on CompareNotesEHealth; correct command line instructions - LIMSI 2015

----------------------------------------------------
Entities are evaluated using the following command:

java -cp brateval.jar au.com.nicta.csp.brateval.CompareEntities evaluation_set_folder reference_set_folder exact_match

brateval.jar = adequate version of brateval .jar file
reference_set_folder = reference folder
evaluation_set_folder = folder with annotations to evaluate
exact_match = true - exact match of the entity span / false - overlap between entities span

The entity evaluation results show the statistics for true positives, false negatives and false positives.
Two entities match when they to agree on the entity type and on the span of text (exact or overlap span matches are available).

----------------------------------------------------
Attributes are evaluated using the following command:

java -cp brateval.jar au.com.nicta.csp.brateval.CompareAttributes reference_set_folder evaluation_set_folder exact_match

brateval.jar = adequate version of brateval .jar file
reference_set_folder = reference folder
evaluation_set_folder = folder with annotations to evaluate
exact_match = true - exact match of the entity span / false - overlap between entities span

The attribute evaluation results show the statistics for true positives, false negatives and false positives.
Two attributes match when they to agree on the attribute type and value as well as on the entity they modify (type and span (exact or overlap span matches are available)).

----------------------------------------------------
Relations are evaluated using the following command:

java -cp brateval.jar au.com.nicta.csp.brateval.CompareRelations evaluation_set_folder reference_set_folder exact_match verbose

brateval.jar = adequate version of brateval .jar file
reference_set_folder = reference folder
evaluation_set_folder = folder with annotations to evaluate
exact_match = true - exact match of the entity span / false - overlap between entities span
verbose = true - in addition to the overall comparison statistics, the program shows examples of true positives, false negatives and false positives / false - show only the overall comparison statistics

The relation evaluation results shows the statistics for true positives, false negatives and false positives.
Two relations match when the entities and the relation type match.
The statistics show when the statistics for relations in which the entities are matched in the reference set but the relation does not exist in the reference set.

----------------------------------------------------
AnnotatorNotes are evaluated using the following command:

java -cp brateval.jar au.com.nicta.csp.brateval.CompareNotesEHealth reference_set_folder evaluation_set_folder exact_match

brateval.jar = adequate version of brateval .jar file
reference_set_folder = reference folder
evaluation_set_folder = folder with annotations to evaluate
exact_match = true - exact match of the entity span / false - overlap between entities span

The AnnotatorNotes evaluation results shows the statistics for true positives, false negatives and false positives.
Two AnnotatorNotes match when the entities type, offsets and AnnotatorNotes content match; this is meant for AnnotatorNotes containing either a single CUI or a list of CUIs separated by space. When more than one CUI is provided, a match is effective when at least one of the provided CUIs matches the list of reference CUIs. 


----------------------------------------------------
The initial version of the software has been used to produce results for the Variome corpus presented in the following publication:

Karin Verspoor, Antonio Jimeno Yepes, Lawrence Cavedon, Tara McIntosh, Asha Herten-Crabb, Zoë Thomas, John-Paul Plazzer (2013)
Annotating the Biomedical Literature for the Human Variome.
Database: The Journal of Biological Databases and Curation, virtual issue for BioCuration 2013 meeting. 2013:bat019.
doi:10.1093/database/bat019

The updated version of the software has been used to produce results for a French clincial corpus presented in the following publications:

Deléger L, Grouin C, Ligozat AL, Zweigenbaum P, Névéol A. Annotation of specialized corpora using a comprehensive entity and relation scheme. LREC 2014. 2014:1267-1274.

Névéol A, Grouin C, Tannier X, Hamon T, Kelly L, Goeuriot L, Zweigenbaum P. (2015) Task 1b of the CLEF eHealth Evaluation Lab 2015: Clinical Named Entity Recognition. CLEF 2015 Evaluation Labs and Workshop: Online Working Notes, CEUR-WS, September, 2015.

