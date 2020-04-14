##R Functions for Google Ngrams
This notebook contains functions for processing [Google's Ngram repositories](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html) without having to download them locally. These repositories vary in their size, but the larger ones contain multiple gigabytes. So depending on the specific word forms being searched, loading and processing the data tables can take up to a couple of minutes.

The **google_ngram()** function takes three arguments: **word_forms**, **variety**, and **by**. The first can be a single word like *teenager* or lemmas like *walk*, *walks* and *walked* that are put into a vector: **c("walk", "walks", "walked")**. The same principal applies to ngrams > 1: **c("teenager is", "teenagers are")**. The first word in an ngram sequence should be from the same root. So the function would **fail** to process *c("teenager is", "child is"). The function will combine the counts of all forms in the returned data frame.

The variety argument can be one of: **eng**, **gb**, **us**, or **fiction**, for all English, British English, American English, or fiction, respectively.

The function can also return counts summed and normalized by year or by decade using the by argument: **by="year"** or **by="decade")**.

###Examples

The following would return counts for the word *xray* in US English by year:

`xray_year <- google_ngram(word_forms = "xray", variety = "us", by = "year")`

That result could be plotted using a wrapper function for ggplot:

`plot_year(xray_year)`

Alternatively, the following would return counts of the combined forms *xray* and *xrays* in British English by decade:

`xray_decade <- google_ngram(word_forms = c("xray", "xrays"), variety = "gb", by = "decade")`

That result could be plotted only for the 20th century:

`plot_decade(xray_decade, start = 1900, end = 2000)`
