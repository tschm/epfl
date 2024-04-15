Search.setIndex({"docnames": ["docs/ConditionalValueAtRisk", "docs/ConstructingARestimators", "docs/LeveragedPortfolios", "docs/index"], "filenames": ["docs/ConditionalValueAtRisk.ipynb", "docs/ConstructingARestimators.ipynb", "docs/LeveragedPortfolios.ipynb", "docs/index.md"], "titles": ["The Conditional Value at Risk", "Constructing estimators", "Leveraged Portfolios", "epfl"], "terms": {"http": [0, 1, 2], "en": [0, 1, 2], "wikipedia": [0, 1, 2], "org": [0, 1, 2], "wiki": [0, 1, 2], "expected_shortfal": 0, "import": [0, 1, 2], "numpi": [0, 1, 2], "np": [0, 1, 2], "In": [0, 1], "thi": [0, 2], "talk": [0, 3], "we": [0, 1, 2], "assum": [0, 1], "ar": [0, 1], "postiv": 0, "larger": 0, "more": 0, "pain": 0, "want": [0, 1], "neg": [0, 1, 2], "mathtt": 0, "var": [0, 1], "_": 0, "level": 0, "i": [0, 1, 2, 3], "smallest": 0, "smaller": 0, "than": [0, 1], "doe": 0, "sai": 0, "anyth": 0, "about": 0, "magnitud": 0, "can": [0, 1], "onli": 0, "make": [0, 2], "statement": 0, "number": 0, "n": [0, 1, 2], "1": [0, 1, 2], "ha": [0, 1, 2], "some": [0, 2], "sever": 0, "mathemat": 0, "flaw": 0, "It": [0, 1], "": [0, 1, 2], "sub": 0, "addit": 0, "convex": 0, "broken": 0, "howev": 0, "regul": 0, "embrac": 0, "comput": [0, 1], "mean": 0, "largest": 0, "entri": 0, "vector": [0, 2], "optim": [0, 1], "linear": 0, "combin": 0, "without": 0, "ever": 0, "sort": [0, 1], "ani": 0, "result": [0, 1, 2], "program": 0, "call": [0, 1], "cvar": 0, "an": [0, 1], "upper": 0, "bound": 0, "given": 0, "mathbf": [0, 1, 2], "r": [0, 1, 2], "introduc": [0, 1], "free": [0, 1], "variabl": [0, 1, 2], "gamma": 0, "defin": 0, "function": 0, "f": [0, 1, 2], "begin": [0, 1, 2], "eqnarrai": 0, "frac": [0, 1], "sum": [0, 1, 2], "r_i": 0, "end": [0, 1, 2], "continu": [0, 2], "first": 0, "deriv": 0, "left": 0, "geq": 0, "right": 0, "If": 0, "minim": [0, 1], "particular": 0, "def": [0, 1], "return": [0, 2], "excess": [0, 2], "len": 0, "note": [0, 1, 2], "3": [0, 1, 2], "4": [0, 1], "2": [0, 1, 2], "depend": 0, "your": [0, 2], "definit": 0, "5": [0, 1, 2], "arrai": 0, "x": [0, 1, 2], "linspac": 0, "start": [0, 2], "stop": 0, "num": 0, "1000": 0, "v": 0, "g": [0, 1, 2], "80": 0, "plt": [0, 1], "plot": [0, 1], "grid": 0, "true": [0, 1], "xlabel": [0, 1], "ylabel": [0, 1], "titl": 0, "global": 0, "minimum": 0, "axi": [0, 1], "6": [0, 1], "show": [0, 1], "befor": [0, 1], "us": [0, 1, 2], "conic": 0, "reformul": 0, "align": [0, 1, 2], "min_": [0, 1], "mathbb": [0, 1, 2], "t": [0, 1, 2], "t_i": 0, "text": [0, 2], "now": 0, "www": 0, "cvxpy": [0, 2], "latest": 0, "tutori": 0, "max": [0, 2], "from": [0, 1, 2], "cvx": [0, 1, 2], "util": [0, 1, 2], "interest": 0, "e": [0, 1, 2], "what": 0, "20": 0, "biggest": 0, "object": [0, 1, 2], "int": 0, "po": [0, 1], "print": [0, 1, 2], "A": [0, 1], "sum_largest": 0, "k": 0, "5000000000000001": 0, "3704392": 0, "opt": [0, 2], "hostedtoolcach": [0, 2], "python": [0, 2], "9": [0, 1, 2], "19": [0, 2], "x64": [0, 2], "lib": [0, 2], "python3": [0, 2], "site": [0, 2], "packag": [0, 2], "reduct": [0, 2], "solver": [0, 2], "solving_chain": [0, 2], "py": [0, 1, 2], "336": [0, 2], "futurewarn": [0, 2], "problem": [0, 1, 2], "being": [0, 2], "solv": [0, 2], "eco": [0, 2], "default": [0, 2], "clarabel": [0, 2], "instead": [0, 2], "To": [0, 2], "specifi": [0, 2], "explicitli": [0, 2], "cp": [0, 2], "argument": [0, 1, 2], "method": [0, 2], "warn": [0, 2], "ecos_deprecation_msg": [0, 2], "take": 0, "random": [0, 1, 2], "data": [0, 1, 2], "randn": 0, "2500": 0, "100": [0, 1, 2], "m": [0, 1], "shape": [0, 1], "95": 0, "w": [0, 1], "constraint": [0, 2], "obj": 0, "cvar2": 0, "hist": 0, "weight": [0, 1, 2], "bin": 0, "150": 0, "format": [0, 2], "18295793343647432": 0, "18295793343647423": 0, "could": 0, "length": 0, "do": [0, 2], "need": [0, 1], "element": 0, "nor": 0, "know": 0, "practic": 0, "rather": [0, 1], "have": [0, 1], "asset": [0, 1, 2], "try": 0, "find": [0, 1], "correspond": [0, 1], "portfolio": 0, "matplotlib": 1, "pyplot": 1, "style": [1, 2], "ggplot": 1, "ipython": [1, 2], "core": [1, 2], "displai": [1, 2], "html": [1, 2], "contain": [1, 2], "width": [1, 2], "panda": [1, 2], "pd": [1, 2], "tmp": [1, 2], "ipykernel_1864": 1, "2951110500": 1, "deprecationwarn": [1, 2], "deprec": [1, 2], "sinc": [1, 2], "7": [1, 2], "14": [1, 2], "pleas": [1, 2], "autoregressive_model": 1, "veri": 1, "common": 1, "base": 1, "model": 1, "autoregress": 1, "r_t": 1, "sum_": 1, "w_i": 1, "r_": 1, "predict": 1, "unknown": 1, "previou": 1, "attent": 1, "you": 1, "mai": 1, "volatil": [1, 2], "adjust": 1, "appli": 1, "filter": 1, "etc": 1, "how": 1, "pick": 1, "paramet": 1, "partial": 1, "autocorrel": 1, "convolut": 1, "statsmodel": 1, "tsa": 1, "filtertool": 1, "convolution_filt": 1, "nside": 1, "seri": 1, "0": [1, 2], "trendfollow": 1, "posit": [1, 2], "datafram": 1, "pred": 1, "shift": 1, "corr": 1, "nan": 1, "000000": 1, "895788": 1, "190159": 1, "538431": 1, "revers": 1, "good": 1, "idea": 1, "200": 1, "stattool": 1, "st": 1, "gener": 1, "read_csv": 1, "spx_index": 1, "csv": 1, "squeez": 1, "index_col": 1, "parse_d": 1, "pct_chang": 1, "dropna": 1, "let": 1, "pacf": 1, "nlag": 1, "kind": 1, "bar": 1, "typeerror": 1, "traceback": 1, "most": [1, 2], "recent": 1, "cell": 1, "line": 1, "got": 1, "unexpect": 1, "keyword": 1, "The": 1, "trade": 1, "system": 1, "1e6": 1, "std": 1, "profit": 1, "todai": 1, "yesterdai": 1, "cumsum": 1, "time": [1, 2], "exponenti": 1, "decai": 1, "lambda": 1, "where": 1, "suitabl": 1, "scale": 1, "constant": 1, "neq": 1, "everyth": 1, "move": 1, "averag": 1, "wrong": 1, "exp_weight": 1, "power": 1, "rang": [1, 2], "linalg": 1, "norm": [1, 2], "16": 1, "40": 1, "period": 1, "8": 1, "12": 1, "24": 1, "32": 1, "48": 1, "64": 1, "96": 1, "192": 1, "matrix": [1, 2], "each": 1, "column": 1, "index": 1, "arg": [1, 2], "rvert": [1, 2], "lvert_2": 1, "lstsq": 1, "sometim": 1, "don": 1, "mosek": 1, "valu": [1, 2], "provid": 1, "few": 1, "indic": 1, "avoid": 1, "fast": 1, "prefer": 1, "slower": 1, "thei": 1, "induc": 1, "less": 1, "cost": 1, "signal": 1, "here": [1, 2], "lvert": [1, 2], "x_i": [1, 2], "x_": 1, "delta": 1, "lvert_1": 1, "th": 1, "d_i": 1, "diagon": 1, "penalti": 1, "d": 1, "d_": 1, "aw": 1, "rvert_2": 1, "dw": 1, "mean_vari": 1, "diff": 1, "ab": 1, "lamb": 1, "diag": 1, "kei": 1, "t_weight": 1, "15": 1, "figsiz": 1, "30": 1, "10": 1, "track": 1, "histor": 1, "standard": 1, "total": 1, "help": 1, "expens": 1, "see": 1, "lar": 1, "lasso": 1, "possibl": 1, "establish": 1, "rank": 1, "amongst": 1, "them": 1, "robustli": 1, "vertic": 1, "stack": 1, "across": 1, "group": 1, "plotli": 2, "ipykernel_1892": 2, "1686460620": 2, "30_fund": 2, "alloc": 2, "capit": 2, "c": 2, "sell": 2, "short": 2, "financ": 2, "long": 2, "univers": 2, "max_": 2, "mu": 2, "sigma": 2, "leq": 2, "2c": 2, "sqrt": 2, "sigma_": 2, "maxim": 2, "cov": 2, "expect": 2, "ones": 2, "ey": 2, "05": 2, "sin": 2, "sigma_max": 2, "quad_form": 2, "all": 2, "299999989118021": 2, "299999989118057": 2, "999999999999964": 2, "express": 2, "621": 2, "userwarn": 2, "multipl": 2, "been": 2, "scalar": 2, "multipli": 2, "elementwis": 2, "code": 2, "path": 2, "hit": 2, "so": 2, "far": 2, "msg": 2, "two": 2, "part": 2, "gave": 3, "onc": 3}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"The": 0, "condit": 0, "valu": 0, "risk": 0, "alpha": 0, "0": 0, "99": 0, "tail": 0, "loss": 0, "distribut": 0, "summari": [0, 1, 2], "construct": 1, "estim": 1, "thoma": [1, 2], "schmelzer": [1, 2], "look": 1, "onli": 1, "last": 1, "two": 1, "return": 1, "might": 1, "bit": 1, "bia": 1, "naiv": 1, "regress": 1, "mean": 1, "variat": 1, "leverag": 2, "portfolio": 2, "A": 2, "130": 2, "30": 2, "equiti": 2, "epfl": 3}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 60}, "alltitles": {"The Conditional Value at Risk": [[0, "the-conditional-value-at-risk"]], "The \\alpha=0.99 tail of a loss distribution": [[0, "the-alpha-0-99-tail-of-a-loss-distribution"]], "Summary": [[0, "summary"], [1, "summary"], [2, "summary"]], "Constructing estimators": [[1, "constructing-estimators"]], "Thomas Schmelzer": [[1, "thomas-schmelzer"], [2, "thomas-schmelzer"]], "Looking only at the last two returns might be a bit \u2026": [[1, "looking-only-at-the-last-two-returns-might-be-a-bit"]], "Bias": [[1, "bias"]], "(Naive) regression": [[1, "naive-regression"]], "Mean variation": [[1, "mean-variation"]], "Leveraged Portfolios": [[2, "leveraged-portfolios"], [2, "id1"]], "A 130/30 Equity Portfolio": [[2, "a-130-30-equity-portfolio"]], "epfl": [[3, "epfl"]]}, "indexentries": {}})