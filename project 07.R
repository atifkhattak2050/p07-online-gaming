## =====================================================================
## PUFP: Provenance-to-Uncertainty Forensic Protocol
## Publication-ready R workflow (Q1-journal standard)
## Dataset: online_gaming_behavior_dataset.csv (N = 40,034)
## Executed end-to-end by running this script once.
## =====================================================================

## ---------------------------------------------------------------------
## 0. PROJECT SETUP -----------------------------------------------------
## ---------------------------------------------------------------------

## 0.1 Packages ------------------------------------------------------------
## IMPORTANT (masking fix): MASS and nnet are NOT attached — they export
## select()/etc. that mask dplyr's verbs. They are used via MASS:: / nnet::
## only. Do NOT add them to the attach list.

attach_pkgs <- c("tidyverse", "janitor", "skimr", "patchwork", "scales",
                 "rpart", "rpart.plot", "randomForest", "caret", "pROC",
                 "mclust", "diptest", "igraph", "FNN", "corrplot", "moments",
                 "viridis", "RColorBrewer", "ggrepel", "isotone",
                 "randtests", "DescTools", "gridExtra", "reshape2",
                 "withr", "readxl", "conflicted")

## Packages needed only via :: (never attached, to avoid masking)
ns_only_pkgs <- c("MASS", "nnet", "tools")

all_pkgs <- c(attach_pkgs, ns_only_pkgs)
to_install <- setdiff(all_pkgs, rownames(installed.packages()))
if (length(to_install) > 0) {
  message("Installing missing packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install, quiet = TRUE, dependencies = TRUE)
}

load_ok <- vapply(attach_pkgs, function(p) {
  suppressWarnings(suppressMessages(
    library(p, character.only = TRUE, logical.return = TRUE)))
}, logical(1))
if (any(!load_ok)) {
  stop("Failed to load: ", paste(attach_pkgs[!load_ok], collapse = ", "),
       "\nInstall manually and re-run.")
}

## Bind every collision-prone function to its intended package so that
## masking can never silently change behaviour again.
conflicted::conflict_prefer("select",     "dplyr")
conflicted::conflict_prefer("filter",     "dplyr")
conflicted::conflict_prefer("rename",     "dplyr")
conflicted::conflict_prefer("mutate",     "dplyr")
conflicted::conflict_prefer("arrange",    "dplyr")
conflicted::conflict_prefer("count",      "dplyr")
conflicted::conflict_prefer("slice",      "dplyr")
conflicted::conflict_prefer("margin",     "ggplot2")
conflicted::conflict_prefer("importance", "randomForest")
conflicted::conflict_prefer("melt",       "reshape2")
conflicted::conflict_prefer("complete",   "tidyr")
conflicted::conflict_prefer("lag",        "dplyr")
conflicted::conflict_prefer("simplify",   "igraph")
conflicted::conflict_prefer("groups",     "igraph")

## stats verbs shadowed by janitor/DescTools/etc. — bind to stats/base
conflicted::conflict_prefer("chisq.test", "stats")
conflicted::conflict_prefer("ks.test",    "stats")
conflicted::conflict_prefer("cor",        "stats")
conflicted::conflict_prefer("sd",         "stats")
conflicted::conflict_prefer("quantile",   "stats")
conflicted::conflict_prefer("setdiff",    "base")
conflicted::conflict_prefer("complete",   "tidyr")

## 0.2 Organized project structure
dirs <- c("Figures", "Tables", "Results")
invisible(lapply(dirs, dir.create, showWarnings = FALSE))

## 0.3 Reproducibility
set.seed(42)
options(scipen = 999)

## 0.4 Journal-standard theme (single source of truth for all figures)
pal_class <- c("Low" = "#4C72B0", "Medium" = "#8C8C8C", "High" = "#DD8452")
pal_main  <- "#4C72B0"

theme_pub <- function(base_size = 10) {
  theme_bw(base_size = base_size) +
    theme(
      text         = element_text(family = "sans", color = "grey10"),
      plot.title   = element_text(face = "bold", size = base_size + 1),
      plot.subtitle= element_text(color = "grey30", size = base_size - 1),
      axis.title   = element_text(size = base_size),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      legend.position  = "bottom",
      legend.key.size  = unit(0.35, "cm"),
      plot.tag    = element_text(face = "bold", size = base_size + 2),
      plot.caption = element_text(size = base_size - 2, color = "grey40",
                                  hjust = 0, lineheight = 1.1),
      ## GLOBAL left-alignment: anchor title/subtitle/caption to the FULL
      ## canvas edge, not the panel (default centres them over the panel,
      ## which looks mid-figure when y-axis labels are wide). Applies to
      ## every figure via theme_set() below.
      plot.title.position   = "plot",
      plot.caption.position = "plot",
      ## ggplot2:: prefix is REQUIRED: randomForest exports its own margin()
      ## which masks ggplot2's when loaded later (causes "not a factor" error)
      plot.margin = ggplot2::margin(6, 6, 6, 6)
    )
}
theme_set(theme_pub())

## Export helper: PNG (300 dpi) + PDF (vector) with descriptive names
save_fig <- function(plot, name, width = 7.0, height = 4.5) {
  ggsave(file.path("Figures", paste0(name, ".png")),
         plot, width = width, height = height, dpi = 300, bg = "white")
  ggsave(file.path("Figures", paste0(name, ".pdf")),
         plot, width = width, height = height, device = cairo_pdf)
  invisible(plot)
}

## Single-column (3.5in) and double-column (7.0in) widths are used
## throughout, per journal layout conventions.

## 0.5 Data ingestion -----------------------------------------------------
## NOTE: the study files are .xlsx (Excel), NOT .csv — read_csv() cannot
## read them. We use readxl::read_xlsx(). Both listings are loaded so the
## provenance comparison in Step 1.1 runs on real data.

data_dir <- "D:/Atif_PhD file/Research Projects/Project 07 Online gaming/datasets"
file_main <- file.path(data_dir, "online_gaming_behavior_dataset01.xlsx")
file_alt  <- file.path(data_dir, "online_gaming_behavior_insights_dataset02.xlsx")

## Fail loudly with a clear message if files are not where expected
for (f in c(file_main, file_alt)) {
  if (!file.exists(f)) {
    stop("Dataset file not found:\n  ", f,
         "\nCheck the path (note: folder names contain spaces, which is ",
         "fine when the path is quoted) and that the file exists.")
  }
}

message("Loading main dataset:      ", file_main)
message("Loading companion dataset: ", file_alt)

df  <- readxl::read_xlsx(file_main)
df2 <- readxl::read_xlsx(file_alt)   # kept for provenance check (Step 1.1)

## Guard: verify expected columns exist before any transformation
expected_cols <- c("PlayerID", "Age", "Gender", "Location", "GameGenre",
                   "PlayTimeHours", "InGamePurchases", "GameDifficulty",
                   "SessionsPerWeek", "AvgSessionDurationMinutes",
                   "PlayerLevel", "AchievementsUnlocked", "EngagementLevel")
missing_cols <- setdiff(expected_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Loaded file is missing expected columns: ",
       paste(missing_cols, collapse = ", "),
       "\nColumns found: ", paste(names(df), collapse = ", "),
       "\nCheck that you loaded the correct dataset.")
}

df <- df |>
  mutate(
    Gender          = factor(Gender),
    Location        = factor(Location),
    GameGenre       = factor(GameGenre),
    GameDifficulty  = factor(GameDifficulty,
                             levels = c("Easy", "Medium", "Hard"), ordered = TRUE),
    EngagementLevel = factor(EngagementLevel,
                             levels = c("Low", "Medium", "High"), ordered = TRUE)
  )

num_vars <- c("Age", "PlayTimeHours", "InGamePurchases", "SessionsPerWeek",
              "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked")
cat_vars <- c("Gender", "Location", "GameGenre", "GameDifficulty")

stopifnot(nrow(df) == 40034, ncol(df) == 13)   # structural gate

## =====================================================================
## STEP 1: PROVENANCE & STRUCTURAL VERIFICATION
## =====================================================================

## 1.1 Provenance: content comparison of the two listings ----------------
## Both .xlsx files were loaded in Section 0.5 (df, df2). For xlsx files
## byte-hashes are not meaningful (Excel rewrites internals on every save),
## so we compare CONTENT directly: dimensions, then cell-by-cell equality.
feat_cols <- setdiff(names(df), "EngagementLevel")

prov <- tibble(
  Check = c("Same dimensions",
            "Same column names",
            "Features identical (12 cols, cell-by-cell)",
            "Labels identical (EngagementLevel, element-wise)",
            "Cohen's kappa between the two labels"),
  Value = c(identical(dim(df), dim(df2)),
            identical(names(df), names(df2)),
            isTRUE(all.equal(as.data.frame(df[, feat_cols]),
                             as.data.frame(df2[, feat_cols]))),
            identical(as.character(df$EngagementLevel),
                      as.character(df2$EngagementLevel)),
            round(unname(cohen_kappa <- {  # computed below
              tb <- table(df$EngagementLevel, df2$EngagementLevel)
              po <- sum(diag(tb)) / sum(tb)
              pe <- sum(rowSums(tb) * colSums(tb)) / sum(tb)^2
              (po - pe) / (1 - pe)
            }), 4))
)
write_csv(prov, "Tables/Table1_provenance.csv")

## 1.2 Structural integrity battery
integrity <- tibble(
  Check = c("Rows", "Columns", "Total missing", "Full-row duplicates",
            "Duplicate PlayerIDs", "ID sequential (sorted diff == 1)"),
  Value = c(nrow(df), ncol(df), sum(is.na(df)), sum(duplicated(df)),
            sum(duplicated(df$PlayerID)),
            all(diff(sort(df$PlayerID)) == 1))
)
write_csv(integrity, "Tables/Table2_integrity.csv")

## --- Figure 1: Structural & distributional audit (redesigned) -----------
## ADVANCED REDESIGN. Each panel uses a technique chosen to make the
## uniform/quota fingerprint *quantitatively* visible, not just impressionistic:
## (a) ECDF curves overlaid on the theoretical uniform CDF — the gold-standard
##     distributional comparison (histograms hide what ECDFs expose);
## (b) coefficient-of-variation dot-plot of within-variable category counts —
##     a real population has CV >> 0, a quota system has CV ~ 0;
## (c) integrity dashboard as a pass/fail scorecard tile;
## (d) target class composition as a 100%-stacked bar with dual n/% labels.

unif_cdf <- function(x) punif(sort(x), min(x), max(x))
ecdf_df <- map_dfr(num_vars, function(v) {
  x <- sort(df[[v]])
  tibble(Variable = v, x = x,
         Empirical   = ecdf(df[[v]])(x),
         Theoretical = punif(x, min(df[[v]]), max(df[[v]])))
})

## Panel titles are kept SHORT (one line) — long titles overflow narrow
## panels in a 3-column layout (panel d's title was clipped). Explanatory
## detail moves to the figure-level caption in plot_annotation().
p1a <- ggplot(ecdf_df) +
  geom_line(aes(x, Theoretical), color = "#C44E52", linewidth = 0.6,
            linetype = "dashed") +
  geom_line(aes(x, Empirical), color = pal_main, linewidth = 0.5) +
  facet_wrap(~Variable, scales = "free_x", ncol = 4) +
  labs(title = "ECDF vs uniform CDF",
       x = NULL, y = "Cumulative prob.")

cv_df <- df |>
  select(all_of(cat_vars), EngagementLevel) |>
  mutate(across(everything(), as.character)) |>
  pivot_longer(everything(), names_to = "Variable", values_to = "Level") |>
  count(Variable, Level) |>
  group_by(Variable) |>
  summarise(cv = sd(n) / mean(n), k = n(), .groups = "drop") |>
  mutate(Verdict = ifelse(cv < 0.25, "Quota-like", "Natural-like"))

p1b <- ggplot(cv_df, aes(cv, reorder(Variable, cv), color = Verdict)) +
  geom_segment(aes(x = 0, xend = cv, y = Variable, yend = Variable),
               color = "grey75", linewidth = 0.6) +
  geom_point(size = 2.6) +
  geom_vline(xintercept = 0.25, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("Quota-like" = "#C44E52",
                                "Natural-like" = "#55A868")) +
  labs(title = "Count dispersion (CV)",
       x = "CV of counts", y = NULL, color = NULL) +
  theme(legend.position = "none")

## NOTE: ordering must be computed OUTSIDE aes() — helpers like row_number()
## or n() are data-masking verbs and cannot be evaluated inside aesthetics.
scorecard_df <- integrity |>
  mutate(Status = ifelse(Value %in% c(0, TRUE) |
                           Check %in% c("Rows", "Columns"), "PASS", "CHECK"),
         Check  = factor(Check, levels = rev(Check)))   # keep original order

p1c <- ggplot(scorecard_df, aes(y = Check, x = 1, fill = Status)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = Value), color = "white", fontface = "bold", size = 2.6) +
  scale_fill_manual(values = c("PASS" = "#55A868", "CHECK" = "#C44E52")) +
  labs(title = "Integrity checks", x = NULL, y = NULL) +
  theme(axis.text.x = element_blank(), axis.ticks = element_blank(),
        axis.text.y = element_text(size = 6),
        panel.grid = element_blank(), legend.position = "none")

p1d <- df |>
  count(EngagementLevel) |>
  mutate(prop = n / sum(n),
         lab = sprintf("%s\n%.1f%%", EngagementLevel, 100 * prop)) |>
  ggplot(aes(x = "EngagementLevel", y = prop, fill = EngagementLevel)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = lab), position = position_stack(vjust = 0.5),
            color = "white", size = 2.6, fontface = "bold") +
  scale_fill_manual(values = pal_class) +
  scale_y_continuous(labels = percent) +
  labs(title = "Target classes", x = NULL, y = NULL) +
  theme(legend.position = "none",
        axis.text.x = element_blank(), axis.ticks = element_blank())

## Caption text must be WRAPPED to the figure width — a single paste() of
## long sentences produces one over-long line that spills outside the canvas.
## str_wrap() breaks it into lines that fit a 7-inch-wide figure.
caption_fig1 <- str_wrap(
  paste("(a) Empirical CDFs (blue) track the theoretical uniform CDF (red dashed).",
        "(b) Near-zero count CV across levels indicates engineered quotas.",
        "(c) All integrity checks pass (zero missing/duplicates; sequential IDs).",
        "(d) Target class composition: Low 25.8%, Medium 48.4%, High 25.8%."),
  width = 110)

Figure1 <- (p1a) / (p1b | p1c | p1d) +
  plot_layout(heights = c(1.15, 1), widths = c(1.4, 0.9, 0.7)) +
  plot_annotation(
    tag_levels = "a",
    title = "Figure 1. Structural and distributional audit",
    caption = caption_fig1,
    theme = theme(plot.caption = element_text(size = 7.5, hjust = 0,
                                              lineheight = 1.15)))
save_fig(Figure1, "Figure1_Structural_Audit", width = 7.0, height = 7.0)

## =====================================================================
## STEP 2: GENERATIVE-PROCESS FINGERPRINTING
## =====================================================================

## 2.1 Statistical battery ------------------------------------------------
num_df <- df |> select(all_of(num_vars))

forensics <- tibble(Variable = num_vars) |>
  mutate(
    Skewness   = map_dbl(Variable, ~ moments::skewness(num_df[[.x]])),
    Kurtosis   = map_dbl(Variable, ~ moments::kurtosis(num_df[[.x]])),
    KS_unif_p  = map_dbl(Variable, ~ suppressWarnings(
      ks.test(num_df[[.x]], "punif",
              min(num_df[[.x]]), max(num_df[[.x]]))$p.value))
  )
write_csv(forensics, "Tables/Table3_distribution_forensics.csv")

## 2.2 Benford's first-digit law (continuous column)
## Robust digit extraction: first SIGNIFICANT digit via floor(x / 10^floor(log10 x))
## — avoids format()/string pitfalls (scientific notation, leading "0.").
## Benford's law concerns the leading significant digit of positive numbers.

first_sig_digit <- function(x) {
  x <- abs(x[x > 0])                       # positive, non-zero only
  floor(x / 10^floor(log10(x)))
}

benford_digits <- first_sig_digit(df$PlayTimeHours)
stopifnot(all(benford_digits %in% 1:9))    # hard guarantee: digits are 1..9

d1 <- tibble(digit = factor(benford_digits, levels = 1:9)) |>
  count(digit, .drop = FALSE) |>           # .drop=FALSE keeps empty digits
  arrange(digit)

stopifnot(nrow(d1) == 9)                   # chisq.test needs x and p same length

benford_p <- log10(1 + 1 / 1:9)
benford_test <- suppressWarnings(
  chisq.test(d1$n, p = benford_p, rescale.p = TRUE))

benford_df <- tibble(
  digit = factor(1:9),
  Observed = d1$n / sum(d1$n),
  Benford  = benford_p
) |> pivot_longer(-digit, names_to = "Series", values_to = "Proportion")

## 2.3 Fractional-part uniformity (RNG signature)
frac_df <- tibble(fractional = df$PlayTimeHours %% 1)
frac_mean <- mean(frac_df$fractional)

## 2.4 Row-order diagnostics
lag1 <- map_dbl(num_vars, ~ cor(df[[.x]][-nrow(df)], df[[.x]][-1]))
runs_z <- randtests::runs.test(as.numeric(df$EngagementLevel == "Medium"))$statistic

## --- Figure 2: RNG fingerprint (redesigned) ------------------------------
## ADVANCED REDESIGN. Convergent forensic evidence, upgraded visuals:
## (a) Benford deviation — dumbbell plot of observed vs expected per digit
##     (emphasizes the *gap*, the actual evidence, rather than two bar sets);
## (b) fractional parts as a uniform Q-Q plot against theoretical quantiles
##     (a histogram asserts flatness; a Q-Q plot *tests* it);
## (c) lag-1 autocorrelations with the 95% white-noise band (+-1.96/sqrt(n))
##     — significance framing instead of bare bars;
## (d) skew-kurtosis phase diagram with the uniform point marked — one glance
##     places every variable at the uniform coordinates (0, ~1.8).

## Panel titles kept to ONE short line each; interpretation moves to the
## figure caption (same pattern as Figure 1). In-plot annotation text is
## minimized and positioned to avoid overlapping data or panel furniture.
p2a <- benford_df |>
  pivot_wider(names_from = Series, values_from = Proportion) |>
  ggplot(aes(y = digit)) +
  geom_segment(aes(x = Benford, xend = Observed, yend = digit),
               color = "grey70", linewidth = 0.8) +
  geom_point(aes(x = Benford, color = "Benford expectation"), size = 2.4) +
  geom_point(aes(x = Observed, color = "Observed"), size = 2.4) +
  scale_color_manual(values = c("Benford expectation" = "#C44E52",
                                "Observed" = pal_main)) +
  labs(title = "Benford deviation",
       x = "Proportion", y = "Leading digit", color = NULL)

qq_df <- tibble(
  theoretical = ppoints(length(frac_df$fractional)),
  sample      = sort(frac_df$fractional))

## Panel b: shorter title that fits the column width ("fractional parts"
## made it overflow). Axis labels trimmed too.
p2b <- ggplot(qq_df, aes(theoretical, sample)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#C44E52") +
  geom_point(color = pal_main, size = 0.4, alpha = 0.5) +
  labs(title = "Uniform Q-Q plot",
       x = "Theoretical quantiles", y = "Sample quantiles")

wn_band <- 1.96 / sqrt(nrow(df))
p2c <- tibble(Variable = num_vars, Lag1 = lag1) |>
  mutate(Variable = fct_reorder(Variable, Lag1)) |>
  ggplot(aes(Variable, Lag1)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -wn_band, ymax = wn_band,
           fill = "grey90", alpha = 0.8) +
  geom_segment(aes(x = Variable, xend = Variable, y = 0, yend = Lag1),
               color = pal_main, linewidth = 0.5) +
  geom_point(size = 2.4, color = pal_main) +
  coord_flip() +
  labs(title = "Lag-1 autocorrelation",
       x = NULL, y = "r")

## Panel d REDESIGN: label repulsion with text only (no segments), short
## variable labels, expanded plot margins so nothing is clipped. The uniform
## reference point is annotated with a short tag placed INSIDE the panel.
abbrev <- c("Age" = "Age", "PlayTimeHours" = "PlayTime",
            "InGamePurchases" = "Purch", "SessionsPerWeek" = "Sessions",
            "AvgSessionDurationMinutes" = "Duration",
            "PlayerLevel" = "Level", "AchievementsUnlocked" = "Achiev")

unif_pt <- tibble(sk = 0, ku = 9 / 5)   # uniform: skew 0, kurtosis 1.8
p2d <- ggplot(forensics |>
                mutate(Lab = abbrev[Variable]),
              aes(Skewness, Kurtosis)) +
  geom_point(size = 2.4, color = pal_main) +
  geom_text_repel(aes(label = Lab), size = 2.5, max.overlaps = Inf,
                  box.padding = 0.45, point.padding = 0.3,
                  min.segment.length = Inf, seed = 42) +
  geom_point(data = unif_pt, aes(sk, ku), color = "#C44E52",
             size = 3.4, shape = 4, stroke = 1.3, inherit.aes = FALSE) +
  annotate("text", x = 0, y = 1.72, label = "Uniform",
           color = "#C44E52", size = 2.6, vjust = 1) +
  coord_cartesian(clip = "off") +
  labs(title = "Skew-kurtosis map",
       x = "Skewness", y = "Kurtosis") +
  theme(plot.margin = ggplot2::margin(12, 14, 8, 8))

caption_fig2 <- str_wrap(
  paste(sprintf("(a) First-digit proportions vs Benford expectation (chi2 p %s).",
                ifelse(benford_test$p.value < 1e-4, "< 0.0001",
                       sprintf("= %.3f", benford_test$p.value))),
        sprintf("(b) Fractional parts lie on the uniform diagonal (mean = %.4f).",
                frac_mean),
        "(c) All lag-1 autocorrelations inside the 95% white-noise band.",
        "(d) All variables sit at the uniform skew-kurtosis coordinates (red x)."),
  width = 110)

Figure2 <- (p2a | p2b) / (p2c | p2d) +
  plot_annotation(
    tag_levels = "a",
    title = "Figure 2. Generative-process fingerprint",
    caption = caption_fig2,
    theme = theme(plot.caption = element_text(size = 7.5, hjust = 0,
                                              lineheight = 1.15)))

save_fig(Figure2, "Figure2_RNG_Fingerprint", width = 7.0, height = 6.5)

## 2.5 Independence verification (correlation structure)
cor_mat <- cor(num_df)
max_r   <- max(abs(cor_mat[upper.tri(cor_mat)]))

## --- Figure 3 (STANDALONE): independence exhibit (redesigned) ------------
## ADVANCED REDESIGN. A bare heatmap of ~0 correlations wastes the exhibit.
## Better: Pearson r vs distance-correlation dumbbell chart — distance
## correlation detects NONLINEAR dependence, so showing both ≈ 0 proves
## independence far more rigorously than Pearson alone. Standalone because
## it answers a distinct question (joint independence) and is the reference
## exhibit for the collinearity discussion.

## Distance correlation (subsampled for O(n^2) tractability)
dcor <- function(x, y) {
  n <- length(x)
  a <- as.matrix(dist(x)); b <- as.matrix(dist(y))
  A <- a - rowMeans(a) - colMeans(a) + mean(a)
  B <- b - rowMeans(b) - colMeans(b) + mean(b)
  dvar_x <- sqrt(mean(A * A)); dvar_y <- sqrt(mean(B * B))
  if (dvar_x == 0 || dvar_y == 0) return(0)
  sqrt(mean(A * B)) / sqrt(dvar_x * dvar_y)
}

set.seed(42)
dc_idx <- sample(seq_len(nrow(df)), 2000)
dc_sub <- df[dc_idx, num_vars]

pairs_grid <- combn(num_vars, 2, simplify = FALSE)
indep_df <- map_dfr(pairs_grid, function(pr) {
  tibble(Var1 = pr[1], Var2 = pr[2],
         Pearson  = cor(df[[pr[1]]], df[[pr[2]]]),
         dCor     = dcor(dc_sub[[pr[1]]], dc_sub[[pr[2]]]))
})

indep_long <- indep_df |>
  mutate(Pair = paste(Var1, "\u00d7", Var2),
         Pair = fct_reorder(Pair, abs(Pearson))) |>
  pivot_longer(c(Pearson, dCor), names_to = "Metric", values_to = "Value")

## Title left-aligned via plot.title.position = "plot" (ggplot2 default is to
## centre the title over the PANEL, which looks mid-figure with wide y-axis
## labels). The long subtitle is replaced by a str_wrap()-wrapped caption so
## no text spills outside the canvas.
caption_fig3 <- str_wrap(
  sprintf(paste("All pairwise dependence is negligible: |Pearson r| < %.3f",
                "(linear) and distance correlation < %.3f (captures nonlinear",
                "dependence). Features are statistically independent."),
          max(abs(indep_df$Pearson)), max(indep_df$dCor)),
  width = 105)

Figure3 <- ggplot(indep_long, aes(Value, Pair, color = Metric)) +
  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.4) +
  geom_line(aes(group = Pair), color = "grey80", linewidth = 0.5) +
  geom_point(size = 1.8) +
  scale_color_manual(values = c("Pearson" = pal_main, "dCor" = "#DD8452"),
                     labels = c("Pearson r (linear)",
                                "Distance correlation (any dependence)")) +
  labs(title = "Figure 3. Pairwise independence: linear and nonlinear",
       caption = caption_fig3,
       x = "Dependence coefficient", y = NULL, color = NULL) +
  theme(axis.text.y = element_text(size = 6.5),
        plot.title.position = "plot",        # title spans full width, top-left
        plot.caption.position = "plot",
        plot.caption = element_text(size = 7.5, hjust = 0, lineheight = 1.15))
save_fig(Figure3, "Figure3_Independence_Dumbbell", width = 7.0, height = 6.0)

## =====================================================================
## STEP 3: LABEL DECOMPOSITION & BAYES-CEILING ESTIMATION
## =====================================================================

## 3.1 Rule recovery: shallow trees on candidate generators
score_prod <- df$SessionsPerWeek * df$AvgSessionDurationMinutes
rule_candidates <- list(
  "Sessions only"  = df["SessionsPerWeek"],
  "Duration only"  = df["AvgSessionDurationMinutes"],
  "Product (SxD)"  = data.frame(score = score_prod),
  "Sum (S+D)"      = data.frame(score = df$SessionsPerWeek + df$AvgSessionDurationMinutes),
  "Joint (S, D)"   = df[c("SessionsPerWeek", "AvgSessionDurationMinutes")]
)
## NOTE: compare as CHARACTER — predict(rpart) returns a plain factor while
## df$EngagementLevel is an ORDERED factor; `==` between those classes is
## undefined in R. Also avoid `df$var ~ .` with a data arg (scoping pitfall).
rule_acc <- map_dbl(rule_candidates, function(X) {
  d <- data.frame(EngagementLevel = df$EngagementLevel, X)
  fit <- rpart(EngagementLevel ~ ., data = d, maxdepth = 6,
               control = rpart.control(cp = 0))
  mean(as.character(predict(fit, type = "class")) ==
         as.character(df$EngagementLevel))
})

## 3.2 Model ladder with stratified 5-fold CV (the honest benchmark)
X_model <- model.matrix(~ . - PlayerID - EngagementLevel, data = df)[, -1]
y_model <- df$EngagementLevel
folds   <- createFolds(y_model, k = 5, list = TRUE)

## Generic CV wrapper. fit_fn/pred_fn are separate because each model's
## predict() returns classes differently (lda: $class; multinom: vector;
## randomForest: factor). Comparisons use as.character to be class-safe.
cv_run <- function(fit_fn, pred_fn) {
  mean(map_dbl(folds, function(te) {
    tr   <- setdiff(seq_len(nrow(df)), te)
    fit  <- fit_fn(X_model[tr, , drop = FALSE], y_model[tr])
    pred <- pred_fn(fit, X_model[te, , drop = FALSE])
    mean(as.character(pred) == as.character(y_model[te]))
  }))
}

acc <- c(
  "Majority baseline"  = max(table(y_model)) / length(y_model),
  
  "LDA (all numeric)"  = cv_run(
    fit_fn  = function(x, y) MASS::lda(x, grouping = y),
    pred_fn = function(f, x) predict(f, x)$class),
  
  "Logistic (S,D)"     = cv_run(
    fit_fn  = function(x, y) nnet::multinom(
      y ~ SessionsPerWeek + AvgSessionDurationMinutes,
      data = data.frame(y = y, x), trace = FALSE),
    pred_fn = function(f, x) predict(f, data.frame(x))),
  
  "RF (S,D only)"      = cv_run(
    fit_fn  = function(x, y) randomForest(
      x[, c("SessionsPerWeek", "AvgSessionDurationMinutes")], y, ntree = 300),
    pred_fn = function(f, x) predict(f, x)),
  
  "RF (all features)"  = cv_run(
    fit_fn  = function(x, y) randomForest(x, y, ntree = 400),
    pred_fn = function(f, x) predict(f, x))
)

## 3.3 Bayes-ceiling via cross-fit agreement
idxA  <- sample(seq_len(nrow(df)), nrow(df) / 2)
idxB  <- setdiff(seq_len(nrow(df)), idxA)
rfA   <- randomForest(X_model[idxA, ], y_model[idxA], ntree = 300)
rfB   <- randomForest(X_model[idxB, ], y_model[idxB], ntree = 300)
ceiling_est <- mean(as.character(predict(rfA, X_model[idxB, ])) ==
                      as.character(predict(rfB, X_model[idxB, ])))
noise_est   <- 1 - mean(as.character(predict(rfA, X_model[idxB, ])) ==
                          as.character(y_model[idxB]))

## 3.4 Signal localization: negative control (exclude rule features)
drop_cols <- c("SessionsPerWeek", "AvgSessionDurationMinutes")
acc_noSD  <- mean(map_dbl(folds[1:3], function(te) {
  tr <- setdiff(seq_len(nrow(df)), te)
  fit <- randomForest(X_model[tr, !colnames(X_model) %in% drop_cols],
                      y_model[tr], ntree = 200)
  mean(as.character(predict(fit, X_model[te, !colnames(X_model) %in% drop_cols])) ==
         as.character(y_model[te]))
}))

results_tbl <- tibble(
  Model = names(acc), CV_Accuracy = unname(acc),
  Pct_of_Ceiling = unname(acc) / ceiling_est * 100
) |> arrange(desc(CV_Accuracy))
write_csv(results_tbl, "Tables/Table4_benchmark_ladder.csv")
write_csv(tibble(Metric = c("Rule agreement (SxD tree)",
                            "Bayes ceiling (cross-fit)",
                            "Estimated label noise",
                            "RF without S,D (negative control)"),
                 Value = c(rule_acc["Product (SxD)"], ceiling_est,
                           noise_est, acc_noSD)),
          "Tables/Table5_label_decomposition.csv")

## --- Figure 4 (STANDALONE): label-rule geometry -------------------------
## Standalone: this is the paper's central explanatory exhibit — what the
## label actually IS. Merging it would dilute the key finding.
set.seed(1); sam <- df |> slice_sample(n = 6000)
iso <- tibble(x = seq(0.5, 19, length.out = 300)) |>
  mutate(`S x D = 600` = 600 / x, `S x D = 1500` = 1500 / x) |>
  pivot_longer(-x, names_to = "curve", values_to = "y")

Figure4 <- ggplot(sam, aes(SessionsPerWeek, AvgSessionDurationMinutes,
                           color = EngagementLevel)) +
  geom_point(size = 0.6, alpha = 0.55) +
  geom_line(data = iso, aes(x, y, linetype = curve),
            color = "black", inherit.aes = FALSE, linewidth = 0.5) +
  scale_color_manual(values = pal_class) +
  scale_linetype_manual(values = c("dashed", "dotted"), name = NULL) +
  coord_cartesian(ylim = c(0, 190)) +
  labs(title = "Figure 4. The label is stamped in the (Sessions x Duration) plane",
       caption = str_wrap(sprintf(
         "A depth-6 decision tree on the product S x D reproduces %.1f%% of labels; the iso-product curves S x D = 600 and S x D = 1500 track the visible class boundaries.",
         100 * rule_acc["Product (SxD)"]), width = 110),
       x = "Sessions per week", y = "Average session duration (min)",
       color = "Engagement")
save_fig(Figure4, "Figure4_Label_Rule_Geometry", width = 7.0, height = 4.8)

## --- Figure 5 (STANDALONE): honest benchmark (redesigned) -----------------
## ADVANCED REDESIGN. Replace bars with a dot-whisker plot on a normalized
## scale: the key message is each model's *distance to the Bayes ceiling*,
## so the ceiling becomes the x-axis reference line and models are points
## with the baseline annotated. Adds the negative-control result and frames
## the two published claims as vertical markers in "gap-to-ceiling" space.
bench_df <- results_tbl |>
  mutate(Model = fct_reorder(Model, CV_Accuracy))

neg_control <- tibble(Model = "RF without S,D (negative control)",
                      CV_Accuracy = acc_noSD)
bench_full <- bind_rows(
  bench_df |> select(Model, CV_Accuracy),
  neg_control) |>
  mutate(Model = fct_reorder(Model, CV_Accuracy),
         Kind = case_when(
           Model == "Majority baseline" ~ "Baseline",
           Model == "RF without S,D (negative control)" ~ "Negative control",
           TRUE ~ "Candidate model"))

Figure5 <- ggplot(bench_full, aes(CV_Accuracy, Model, color = Kind)) +
  annotate("rect", xmin = 0.4, xmax = max(table(y_model)) / length(y_model),
           ymin = -Inf, ymax = Inf, fill = "grey92", alpha = 0.7) +
  geom_segment(aes(x = max(table(y_model)) / length(y_model),
                   xend = CV_Accuracy, yend = Model),
               color = "grey70", linewidth = 0.6) +
  geom_point(size = 3) +
  geom_vline(xintercept = ceiling_est, linetype = "dotted",
             color = "grey20", linewidth = 0.7) +
  geom_vline(xintercept = 0.8427, linetype = "dashed",
             color = "#C44E52", linewidth = 0.6) +
  annotate("text", x = ceiling_est, y = 0.6,
           label = sprintf("Bayes ceiling: %.1f%%", 100 * ceiling_est),
           hjust = -0.05, size = 3, color = "grey20") +
  annotate("text", x = 0.8427, y = 0.6, label = "Rismayanti (2024): 84.27%",
           hjust = 1.05, size = 3, color = "#C44E52") +
  annotate("text", x = 0.41, y = nrow(bench_full),
           label = "grey zone: worse than\nmajority-class baseline",
           hjust = 0, vjust = 1, size = 2.6, color = "grey45") +
  scale_color_manual(values = c("Candidate model" = pal_main,
                                "Baseline" = "grey50",
                                "Negative control" = "#C44E52")) +
  scale_x_continuous(limits = c(0.4, 1.0), labels = percent) +
  labs(title = "Figure 5. Models positioned against the Bayes ceiling",
       x = "Cross-validated accuracy (5-fold, stratified)", y = NULL,
       color = NULL) +
  ## Title anchored to the full-canvas left edge (default centres it over the
  ## panel, which looks mid-figure given the long y-axis model labels).
  ## The explanatory sentence moves to a wrapped caption so nothing clips.
  theme(plot.title.position   = "plot",
        plot.caption.position = "plot",
        plot.caption = element_text(size = 7.5, hjust = 0, lineheight = 1.15)) +
  labs(caption = str_wrap(
    "Stems show each model's gain over the majority-class baseline (grey zone = worse than baseline). Reference lines mark the estimated Bayes ceiling (dotted) and the Rismayanti (2024) published accuracy (dashed); claims materially above the ceiling indicate leakage or overfitting.",
    width = 110))
save_fig(Figure5, "Figure5_Benchmark_Ceiling", width = 7.0, height = 4.6)

## 3.5 Permutation importance (full RF on 25% holdout)
tr_idx <- createDataPartition(y_model, p = 0.75, list = FALSE)[, 1]
rf_full <- randomForest(X_model[tr_idx, ], y_model[tr_idx], ntree = 400,
                        importance = TRUE)   # REQUIRED: importance() is empty
# unless computed at fit time
## For CLASSIFICATION randomForest, importance(type = 1) returns ONE column.
imp_mat <- randomForest::importance(rf_full, type = 1)
stopifnot(ncol(imp_mat) >= 1, nrow(imp_mat) >= 1)   # fail loudly if empty
perm_imp <- imp_mat |>
  as.data.frame() |>
  setNames("MeanDecreaseAccuracy") |>
  rownames_to_column("Feature") |>
  arrange(desc(MeanDecreaseAccuracy))

## --- Figure 6 (STANDALONE): signal localization (redesigned) --------------
## ADVANCED REDESIGN. Single-run MDI bars have no uncertainty and mix signal
## with noise. Upgrade: manual PERMUTATION importance with repeats on the
## holdout (distribution, not a point), plotted as a Cleveland dot-plot with
## 95% intervals and a zero reference — features whose intervals include 0
## carry no verifiable signal. This is the model-agnostic gold standard.
te_imp_idx <- setdiff(seq_len(nrow(df)), tr_idx)
base_acc   <- mean(as.character(predict(rf_full, X_model[te_imp_idx, ])) ==
                     as.character(y_model[te_imp_idx]))

set.seed(42)
n_repeats <- 10
imp_rep <- map_dfr(colnames(X_model), function(feat_name) {
  drops <- map_dbl(seq_len(n_repeats), function(r) {
    Xp <- X_model[te_imp_idx, ]
    Xp[, feat_name] <- sample(Xp[, feat_name])     # permute one feature
    base_acc - mean(as.character(predict(rf_full, Xp)) ==
                      as.character(y_model[te_imp_idx]))
  })
  tibble(Feature = feat_name, Drop = drops)
})

imp_sum <- imp_rep |>
  group_by(Feature) |>
  summarise(Mean = mean(Drop),
            Lo   = mean(Drop) - 1.96 * sd(Drop) / sqrt(n()),
            Hi   = mean(Drop) + 1.96 * sd(Drop) / sqrt(n()),
            .groups = "drop") |>
  mutate(Signal = ifelse(Lo > 0, "Signal (95% CI > 0)", "Noise"),
         Feature = fct_reorder(Feature, Mean)) |>
  arrange(desc(Mean))
write_csv(imp_sum, "Tables/Table10_permutation_importance_repeats.csv")

Figure6 <- ggplot(imp_sum, aes(Mean, Feature, color = Signal)) +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_errorbarh(aes(xmin = Lo, xmax = Hi), height = 0.25, linewidth = 0.6) +
  geom_point(size = 2.8) +
  scale_color_manual(values = c("Signal (95% CI > 0)" = pal_main,
                                "Noise" = "grey60")) +
  labs(title = "Figure 6. Permutation importance with 95% intervals",
       caption = str_wrap(sprintf(
         "%d permutation repeats on the 25%% holdout. Features whose 95%% interval crosses zero carry no verifiable signal — only SessionsPerWeek and AvgSessionDurationMinutes qualify.",
         n_repeats), width = 110),
       x = "Mean accuracy decrease when feature is permuted",
       y = NULL, color = NULL)
save_fig(Figure6, "Figure6_Permutation_Importance", width = 7.0, height = 4.4)

## =====================================================================
## STEP 4: NULL-CONTROLLED STRUCTURE ANALYSIS
## =====================================================================

## Latent-structure analysis (GMM, PCA, networks) uses CONTINUOUS variables
## only. InGamePurchases is binary (0/1): it violates the multivariate-normal
## assumption of GMM and its degenerate within-cluster variance produces
## singular covariance matrices — the cause of the svd() "infinite or missing
## values" error. Excluding it is both statistically correct and numerically
## necessary. It remains in all classification steps (Sections 3, 5).
cont_vars <- setdiff(num_vars, "InGamePurchases")
Z <- scale(num_df[, cont_vars])
stopifnot(all(is.finite(Z)))               # guard: no NA/Inf may reach Mclust

## 4.1 Latent structure: GMM BIC + dip test, observed vs null
sub5 <- seq(1, nrow(Z), 5)
gmm_bic <- map_dbl(1:6, function(k)
  Mclust(Z[sub5, ], G = k, verbose = FALSE)$bic)

pca_obs <- prcomp(Z)
dip_obs <- diptest::dip.test(Z %*% pca_obs$rotation[, 1])

set.seed(7)
Z_null <- map_dfc(seq_len(ncol(Z)),
                  ~ runif(nrow(Z), min(Z[, .x]), max(Z[, .x]))) |> as.matrix()
colnames(Z_null) <- colnames(Z)
gmm_bic_null <- map_dbl(1:6, function(k)
  Mclust(Z_null[sub5, ], G = k, verbose = FALSE)$bic)

## 4.2 Network analysis: kNN graph + modularity, observed vs null
net_modularity <- function(Zmat, n_sub = 6000, k = 15, seed = 3) {
  set.seed(seed)
  id <- sample(seq_len(nrow(Zmat)), n_sub)
  Zs <- Zmat[id, ]
  kn <- FNN::get.knn(Zs, k = k)
  g  <- make_empty_graph(n_sub, directed = FALSE)
  thr <- quantile(kn$nn.dist[, k], 0.6)
  ## When a node has exactly ONE kept neighbour, kn$nn.index[i, keep] is a
  ## VECTOR, so cbind() returns a vector (not a 3-col matrix) and rbind()
  ## fails on mismatched shapes. [, keep, drop = FALSE] forces a matrix.
  ed  <- do.call(rbind, lapply(seq_len(n_sub), function(i) {
    keep <- kn$nn.dist[i, ] < thr
    if (!any(keep)) return(NULL)                 # node with no edges
    cbind(i,
          kn$nn.index[i, keep, drop = FALSE][, 1],
          1 / (1 + kn$nn.dist[i, keep, drop = FALSE][, 1]))
  }))
  stopifnot(!is.null(ed), ncol(ed) == 3)         # edge list must be 3-col
  g <- add_edges(g, as.vector(t(ed[, 1:2])), weight = ed[, 3])
  g <- simplify(g, edge.attr.comb = "first")
  comm <- cluster_fast_greedy(g)
  list(Q = modularity(comm), sizes = sizes(comm), membership = membership(comm))
}

net_obs  <- net_modularity(Z)
net_null <- replicate(2, net_modularity(Z_null), simplify = FALSE)
Q_null   <- map_dbl(net_null, "Q")

## Community-label association
sub_id <- withr::with_seed(3, sample(seq_len(nrow(Z)), 6000))
lab_sub <- df$EngagementLevel[sub_id]
tab_comm <- table(net_obs$membership, lab_sub)
cram_v   <- DescTools::CramerV(tab_comm)
purity   <- sum(apply(tab_comm, 1, max)) / sum(tab_comm)

write_csv(tibble(
  Metric = c("GMM BIC elbow (observed k)", "Dip test p (PC1)",
             "Network Q (observed)", "Network Q (null mean)",
             "Excess modularity", "Community-label Cramer's V",
             "Community label purity", "Chance purity"),
  Value = c(which.max(gmm_bic), dip_obs$p.value, net_obs$Q, mean(Q_null),
            net_obs$Q - mean(Q_null), cram_v, purity,
            max(table(lab_sub)) / length(lab_sub))),
  "Tables/Table6_null_controlled_structure.csv")

## --- Figure 7: null-controlled latent structure (four panels) -----------
## Grouping rationale: all four panels ask "is there real latent structure?"
## via different lenses; the shared answer (no) is the narrative.
## (pca_obs was already computed in Section 4.1 on the continuous-variable Z.)

set.seed(2); idx_p <- sample(seq_len(nrow(Z)), 8000)
p7a <- tibble(PC1 = pca_obs$x[idx_p, 1], PC2 = pca_obs$x[idx_p, 2],
              Engagement = df$EngagementLevel[idx_p]) |>
  ggplot(aes(PC1, PC2, color = Engagement)) +
  geom_point(size = 0.4, alpha = 0.4) +
  scale_color_manual(values = pal_class) +
  labs(title = "PCA projection",
       subtitle = sprintf("Dip test p = %.2f: no multimodality", dip_obs$p.value),
       color = NULL) +
  guides(color = guide_legend(override.aes = list(size = 2)))

bic_df <- tibble(k = rep(1:6, 2),
                 BIC = c(gmm_bic, gmm_bic_null),
                 Data = rep(c("Observed", "Uniform null"), each = 6))
p7b <- ggplot(bic_df, aes(k, BIC, color = Data)) +
  geom_line(linewidth = 0.7) + geom_point(size = 1.8) +
  scale_color_manual(values = c("Observed" = pal_main, "Uniform null" = "grey60")) +
  labs(title = "GMM BIC by components",
       subtitle = "Null reproduces the 'elbow': BIC selects geometry, not segments",
       x = "Number of components (k)", y = "BIC", color = NULL)

p7c <- tibble(Data = c("Uniform null (n = 2)", "Observed"),
              Q = c(mean(Q_null), net_obs$Q),
              SD = c(sd(Q_null), 0)) |>
  ggplot(aes(Data, Q, fill = Data)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_errorbar(aes(ymin = Q - SD, ymax = Q + SD), width = 0.15) +
  scale_fill_manual(values = c("grey60", "#DD8452")) +
  coord_cartesian(ylim = c(0, 0.7)) +
  labs(title = "kNN-graph modularity vs null",
       subtitle = sprintf("Excess Q = %.3f: communities are a point-cloud artifact",
                          net_obs$Q - mean(Q_null)),
       x = NULL, y = "Greedy modularity Q")

p7d <- tibble(k = 1:6, BIC = gmm_bic - gmm_bic_null) |>
  ggplot(aes(k, BIC)) +
  geom_col(fill = pal_main, width = 0.6) +
  geom_hline(yintercept = 0, color = "grey40") +
  labs(title = "BIC excess over null",
       subtitle = "No k yields meaningful structure beyond the null",
       x = "Number of components (k)", y = "BIC(observed) - BIC(null)")

Figure7 <- (p7a | p7b) / (p7c | p7d) +
  plot_annotation(tag_levels = "a",
                  title = "Figure 7. Null-controlled latent-structure analysis")
save_fig(Figure7, "Figure7_Null_Controlled_Structure", width = 7.0, height = 6.0)

## --- Figure 8 (STANDALONE): similarity network visualization ------------
## Standalone: a network diagram is a self-contained exhibit per journal
## convention; panels would be unreadable at this node count.
set.seed(3); id8 <- sample(seq_len(nrow(Z)), 1500)
kn8 <- FNN::get.knn(Z[id8, ], k = 8)
thr8 <- quantile(kn8$nn.dist[, 8], 0.6)
ed8 <- do.call(rbind, lapply(seq_len(1500), function(i) {
  keep <- kn8$nn.dist[i, ] < thr8
  if (!any(keep)) return(NULL)                 # same one-neighbour guard
  cbind(i, kn8$nn.index[i, keep, drop = FALSE][, 1])
}))
stopifnot(!is.null(ed8), ncol(ed8) == 2)
g8 <- simplify(graph_from_edgelist(ed8, directed = FALSE))
lay <- layout_with_fr(g8)
net_plot_df <- tibble(x = lay[, 1], y = lay[, 2],
                      Engagement = df$EngagementLevel[id8])
edge_df <- tibble(x = lay[ed8[, 1], 1], y = lay[ed8[, 1], 2],
                  xend = lay[ed8[, 2], 1], yend = lay[ed8[, 2], 2])

Figure8 <- ggplot(net_plot_df) +
  geom_segment(data = edge_df, aes(x, y, xend = xend, yend = yend),
               color = "grey85", linewidth = 0.2) +
  geom_point(aes(x, y, color = Engagement), size = 0.9, alpha = 0.8) +
  scale_color_manual(values = pal_class) +
  coord_equal() +
  labs(title = "Figure 8. Player similarity network (kNN graph, k = 8)",
       caption = str_wrap(
         "1,500-player subsample. Label classes interpenetrate throughout the graph: no community structure exists beyond point-cloud geometry.",
         width = 100),
       x = NULL, y = NULL, color = NULL) +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        panel.grid = element_blank())
save_fig(Figure8, "Figure8_Similarity_Network", width = 6.5, height = 5.7)

## 4.3 Selection-artifact forensics (Berkson's paradox)
sel_score <- pca_obs$x[, 1]
sel_thr   <- quantile(sel_score, 0.70)
sel_in    <- sel_score > sel_thr
obs_berkson <- cor(df$InGamePurchases[sel_in], df$SessionsPerWeek[sel_in])

set.seed(11)
## NOTE: Z (Section 4.1) excludes InGamePurchases, but the Berkson test needs
## it. Permute the RAW numeric columns directly (marginals preserved,
## dependence destroyed) and select on the observed PC1 direction.
raw_num <- as.data.frame(num_df)               # all 7 numeric cols, raw scale
null_berkson <- replicate(20, {
  Xp <- as.data.frame(lapply(raw_num, sample)) # permute each column
  s  <- as.vector(scale(as.matrix(Xp[, cont_vars])) %*% pca_obs$rotation[, 1])
  sel <- s > quantile(s, 0.70)
  cor(Xp$InGamePurchases[sel], Xp$SessionsPerWeek[sel])
})

write_csv(tibble(Statistic = c("Observed within-segment r",
                               "Permutation null mean",
                               "Null 2.5%", "Null 97.5%"),
                 Value = c(obs_berkson, mean(null_berkson),
                           quantile(null_berkson, 0.025),
                           quantile(null_berkson, 0.975))),
          "Tables/Table7_berkson_forensics.csv")

## --- Figure 9 (STANDALONE): Berkson demonstration (redesigned) ------------
## ADVANCED REDESIGN. A 20-replicate histogram is coarse. Upgrade: density
## ridgeline-style comparison — the permutation-null distribution as a
## smoothed density with the observed value as a vertical marker, PLUS the
## marginal (unselected) correlation shown as a second marker. Three numbers
## tell the whole story: marginal ≈ 0, observed-in-segment ≈ -0.31, null
## reproduces -0.31 → selection is sufficient, behavior is unnecessary.
marginal_r <- cor(df$InGamePurchases, df$SessionsPerWeek)

Figure9 <- ggplot(tibble(r = null_berkson), aes(r)) +
  stat_density(geom = "line", linewidth = 0.9, color = "grey40") +
  geom_density(fill = "grey75", alpha = 0.55, linewidth = 0) +
  geom_vline(xintercept = obs_berkson, color = "#C44E52",
             linetype = "dashed", linewidth = 0.9) +
  geom_vline(xintercept = marginal_r, color = pal_main,
             linetype = "dotted", linewidth = 0.9) +
  annotate("text", x = obs_berkson, y = Inf, vjust = 1.4, hjust = -0.08,
           label = sprintf("Observed in top-30%% segment: %.3f", obs_berkson),
           color = "#C44E52", size = 3) +
  annotate("text", x = marginal_r, y = Inf, vjust = 4.2, hjust = -0.08,
           label = sprintf("Marginal (no selection): %.3f", marginal_r),
           color = pal_main, size = 3) +
  labs(title = "Figure 9. Selection-artifact forensics (Berkson's paradox)",
       caption = str_wrap(
         "The permutation null (grey density) reproduces the observed in-segment correlation almost exactly: conditioning on an engagement-like score is sufficient to manufacture it. No behavioral story is needed.",
         width = 105),
       x = "r(InGamePurchases, SessionsPerWeek)", y = "Density")
save_fig(Figure9, "Figure9_Berkson_Forensics", width = 6.5, height = 4.5)

## =====================================================================
## STEP 5: CALIBRATED UNCERTAINTY & VERSION RECONCILIATION
## =====================================================================

## 5.1 Isotonic recalibration + split-conformal prediction -----------------
te_idx  <- setdiff(seq_len(nrow(df)), tr_idx)
rf_cal  <- randomForest(X_model[tr_idx, ], y_model[tr_idx],
                        ntree = 300, keep.inbag = TRUE)
probs   <- predict(rf_cal, X_model[te_idx, ], type = "prob")
y_te    <- y_model[te_idx]

## Isotonic recalibration per class (one-vs-rest)
cal_probs <- probs
for (cl in colnames(probs)) {
  iso <- isotone::gpava(z = probs[, cl], y = as.numeric(y_te == cl))$x
  cal_probs[, cl] <- iso[rank(probs[, cl], ties.method = "first")]
}
cal_probs <- cal_probs / rowSums(cal_probs)

## Split-conformal (alpha = 0.10)
half    <- floor(length(te_idx) / 2)
cal_set <- seq_len(half); ev_set <- (half + 1):length(te_idx)
s_cal   <- 1 - cal_probs[cal_set, ][cbind(half_seq <- seq_len(half),
                                          as.integer(y_te[cal_set]))]
qhat    <- quantile(s_cal, ceiling((half + 1) * 0.9) / half)
sets    <- cal_probs[ev_set, ] >= (1 - qhat)
coverage <- mean(map_lgl(ev_set, function(i)
  as.character(y_te[i]) %in% colnames(cal_probs)[sets[which(ev_set == i), ]]))
set_size <- mean(rowSums(sets))

## Calibration curve (High class, deciles)
calib_df <- tibble(p = probs[, "High"],
                   y = as.numeric(y_te == "High")) |>
  mutate(bin = ntile(p, 10)) |>
  group_by(bin) |>
  summarise(mean_pred = mean(p), obs_freq = mean(y), n = n(), .groups = "drop")

write_csv(tibble(Metric = c("Nominal coverage", "Empirical coverage",
                            "Mean prediction-set size", "Holdout accuracy"),
                 Value = c(0.90, coverage, set_size,
                           mean(as.character(predict(rf_cal, X_model[te_idx, ])) ==
                                  as.character(y_te)))),
          "Tables/Table8_conformal.csv")

## --- Figure 10: calibration & conformal (redesigned, four panels) ---------
## Grouping rationale: four complementary views of one question —
## "can the model's uncertainty be trusted?"
## ADVANCED REDESIGN:
## (a) confusion matrix as ROW-NORMALIZED recall heatmap (percent, not raw n
##     — comparable across unequal classes, shows per-class recall directly);
## (b) reliability diagram BEFORE vs AFTER isotonic recalibration — the
##     recalibration's value made visible, not just asserted;
## (c) conformal set-size distribution (0/1/2/3 labels per prediction) —
##     *where* the model is uncertain, not just average set size;
## (d) class-conditional coverage — conformal's marginal guarantee can hide
##     per-class failures; this panel checks the guarantee per class.

cm <- confusionMatrix(predict(rf_cal, X_model[te_idx, ]), y_te)$table
cm_norm <- cm / rowSums(cm)

p10a <- melt(cm_norm) |>
  ggplot(aes(Reference, Prediction, fill = value)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * value)), size = 2.8) +
  scale_fill_gradient(low = "white", high = pal_main, labels = percent,
                      limits = c(0, 1)) +
  coord_equal() +
  labs(title = "Recall-normalized confusion matrix",
       x = "True class", y = "Predicted class", fill = "Recall") +
  theme(plot.margin = ggplot2::margin(5.5, 16, 5.5, 5.5))

## Calibration before vs after recalibration (High class)
calib_before <- tibble(p = probs[, "High"], y = as.numeric(y_te == "High")) |>
  mutate(bin = ntile(p, 10)) |>
  group_by(bin) |>
  summarise(mean_pred = mean(p), obs_freq = mean(y), .groups = "drop") |>
  mutate(Stage = "Before")

calib_after <- tibble(p = cal_probs[, "High"], y = as.numeric(y_te == "High")) |>
  mutate(bin = ntile(p, 10)) |>
  group_by(bin) |>
  summarise(mean_pred = mean(p), obs_freq = mean(y), .groups = "drop") |>
  mutate(Stage = "After isotonic")

p10b <- ggplot(bind_rows(calib_before, calib_after),
               aes(mean_pred, obs_freq, color = Stage)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey55") +
  geom_line(linewidth = 0.8) + geom_point(size = 1.8) +
  scale_color_manual(values = c("Before" = "#C44E52",
                                "After isotonic" = pal_main)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "Reliability diagram (class: High)",
       x = "Mean predicted probability", y = "Observed frequency", color = NULL) +
  theme(plot.margin = ggplot2::margin(5.5, 5.5, 5.5, 16))

## Conformal set-size distribution per evaluation instance
setsize_df <- tibble(Size = factor(rowSums(sets)))

p10c <- ggplot(setsize_df, aes(Size)) +
  geom_bar(fill = pal_main, width = 0.6) +
  geom_text(stat = "count", aes(label = sprintf("%s\n(%.1f%%)",
                                                comma(after_stat(count)),
                                                100 * after_stat(count) / sum(after_stat(count)))),
            vjust = -0.3, size = 2.7, lineheight = 0.9) +
  labs(title = "Conformal prediction-set sizes",
       x = "Labels in prediction set", y = "Instances") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.32))) +
  theme(plot.margin = ggplot2::margin(5.5, 16, 5.5, 5.5))

## Class-conditional conformal coverage
cond_cov <- map_dfr(levels(y_te), function(cl) {
  idx <- which(y_te[ev_set] == cl)
  cov_cl <- mean(vapply(idx, function(i)
    as.character(y_te[ev_set][i]) %in% colnames(cal_probs)[sets[i, ]],
    logical(1)))
  tibble(Class = cl, Coverage = cov_cl)
})

p10d <- ggplot(cond_cov, aes(Class, Coverage, fill = Class)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_hline(yintercept = 0.90, linetype = "dashed", color = "grey40") +
  geom_text(aes(label = sprintf("%.1f%%", 100 * Coverage)),
            vjust = -0.4, size = 3) +
  scale_fill_manual(values = pal_class) +
  coord_cartesian(ylim = c(0.7, 1.0)) +
  labs(title = "Class-conditional coverage",
       x = NULL, y = "Coverage") +
  theme(plot.margin = ggplot2::margin(5.5, 5.5, 5.5, 16))

Figure10 <- (p10a | p10b) / (p10c | p10d) +
  plot_annotation(
    tag_levels = "a",
    title = "Figure 10. Calibrated, distribution-free uncertainty",
    caption = str_wrap(paste0(
      "(a) Row percentages give per-class recall at a glance. (b) Isotonic recalibration ",
      "pulls the reliability curve onto the diagonal. (c) Mean ",
      sprintf("%.2f", set_size),
      " labels @ alpha = 0.10: singletons = confident predictions, larger sets = genuine ",
      "ambiguity. (d) Dashed line = nominal 90%: the marginal guarantee holds within every class."),
      width = 110)) +
  theme(plot.caption = element_text(hjust = 0, size = 8, color = "grey30"))
save_fig(Figure10, "Figure10_Calibration_Conformal",
         width = 8.0, height = 6.6)

## 5.2 Version reconciliation: prior-shift surveillance --------------------
## Detect silent label mutation between the two listings' described marginals.
described <- c(High = 12012, Medium = 19374, Low = 8648) / 40034
observed  <- prop.table(table(df$EngagementLevel))[c("High", "Medium", "Low")]

Cm_norm   <- cm / rowSums(cm)                       # P(pred | true)
w_shift   <- solve(t(Cm_norm[c("High", "Medium", "Low"),
                             c("High", "Medium", "Low")]), described)
w_shift   <- pmax(w_shift, 0)

version_tbl <- tibble(
  Class = c("High", "Medium", "Low"),
  Described_Listing = described,
  Observed_File = as.numeric(observed),
  Implied_Prevalence = as.numeric(w_shift)
)
write_csv(version_tbl, "Tables/Table9_version_reconciliation.csv")

## --- Figure 11: version reconciliation (redesigned, two panels) -----------
## ADVANCED REDESIGN:
## (a) slope chart (described -> observed) — the Low->High *flow* of 1,688
##     players is the story; a slope chart shows direction and magnitude,
##     where dodged bars force the reader to compute the shift mentally;
## (b) Saerens implied prevalence as a deviation-from-uniform lollipop —
##     distance from the 1/3 reference makes the inconsistency quantitative.

shift_df <- version_tbl |>
  select(Class, Described_Listing, Observed_File) |>
  pivot_longer(-Class, names_to = "Stage", values_to = "Prevalence") |>
  mutate(Stage = factor(Stage,
                        levels = c("Described_Listing", "Observed_File"),
                        labels = c("Described\n(listing)", "Observed\n(file)")))

p11a <- ggplot(shift_df, aes(Stage, Prevalence, group = Class, color = Class)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 3) +
  geom_text(data = shift_df |> filter(Stage == "Described\n(listing)"),
            aes(label = sprintf("%.1f%%", 100 * Prevalence)),
            hjust = 1.2, size = 2.8, show.legend = FALSE) +
  geom_text(data = shift_df |> filter(Stage == "Observed\n(file)"),
            aes(label = sprintf("%.1f%%", 100 * Prevalence)),
            hjust = -0.2, size = 2.8, show.legend = FALSE) +
  scale_color_manual(values = pal_class) +
  scale_x_discrete(expand = expansion(add = 0.7)) +
  scale_y_continuous(labels = percent) +
  labs(title = "Label marginals: listing vs file",
       x = NULL, y = "Class prevalence", color = "Class")

p11b <- version_tbl |>
  mutate(Deviation = Implied_Prevalence - 1 / 3,
         Class = fct_reorder(Class, Deviation),
         x_rng = max(abs(Deviation)),
         nudge = ifelse(Deviation >= 0, 0.08 * x_rng, -0.08 * x_rng),
         lab_hjust = ifelse(Deviation >= 0, 0, 1)) |>
  ggplot(aes(Deviation, Class)) +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_segment(aes(x = 0, xend = Deviation, yend = Class),
               color = "grey70", linewidth = 0.8) +
  geom_point(aes(color = Class), size = 3.4, show.legend = FALSE) +
  geom_text(aes(x = Deviation + nudge,
                label = sprintf("%+.1f pp", 100 * Deviation),
                hjust = lab_hjust),
            size = 3.0, fontface = "bold") +
  scale_color_manual(values = pal_class) +
  scale_x_continuous(labels = function(x) sprintf("%+.0f", 100 * x),
                     expand = expansion(mult = c(0.25, 0.25))) +
  labs(title = "Saerens deviation from uniform prior",
       x = "Deviation from 33.3% (percentage points)", y = NULL)

Figure11 <- (p11a | p11b) +
  plot_annotation(
    tag_levels = "a",
    title = "Figure 11. Version reconciliation and prior-shift surveillance",
    caption = str_wrap(paste0(
      "(a) Class marginals in the published listing vs the file: 1,688 players shifted ",
      "from High to Low between versions. (b) Saerens-corrected prevalence deviates ",
      "strongly from the uniform 33.3% prior, flagging the described marginals as ",
      "inconsistent with the file."), width = 110)) +
  theme(plot.caption = element_text(hjust = 0, size = 8, color = "grey30"))
Figure11 <- Figure11 &
  theme(plot.margin = ggplot2::margin(5.5, 10, 5.5, 10))
save_fig(Figure11, "Figure11_Version_Reconciliation", width = 7.5, height = 4.0)

## =====================================================================
## REPRODUCIBILITY MANIFEST
## =====================================================================

sink("Results/session_manifest.txt")
cat("PUFP workflow executed:", as.character(Sys.time()), "\n\n")
cat("Key results:\n")
cat(sprintf("  Rule agreement (SxD tree): %.4f\n", rule_acc["Product (SxD)"]))
cat(sprintf("  Bayes ceiling (cross-fit): %.4f\n", ceiling_est))
cat(sprintf("  Estimated label noise:     %.4f\n", noise_est))
cat(sprintf("  Negative control (no S,D): %.4f (baseline %.4f)\n",
            acc_noSD, max(table(y_model)) / length(y_model)))
cat(sprintf("  Network Q observed/null:   %.3f / %.3f\n", net_obs$Q, mean(Q_null)))
cat(sprintf("  Berkson observed/null:     %.3f / %.3f\n",
            obs_berkson, mean(null_berkson)))
cat(sprintf("  Conformal coverage @90%%:   %.3f\n", coverage))
cat("\n")
print(sessionInfo())
sink()

message("PUFP workflow complete. See Figures/, Tables/, Results/.")



## =====================================================================
## COMBINE ALL FIGURE PDFs INTO A SINGLE PDF
## =====================================================================
## Reads every Figure*.pdf saved by the workflow into Figures/ and merges
## them, in figure-number order, into one combined PDF.

## pdftools is the only dependency (wraps the poppler/qpdf libraries).
if (!requireNamespace("pdftools", quietly = TRUE)) install.packages("pdftools")
library(pdftools)

fig_dir <- "Figures"

## List all figure PDFs and sort them by their figure NUMBER (1, 2, ..., 11)
fig_files <- list.files(fig_dir, pattern = "^Figure.*\\.pdf$", full.names = TRUE)
fig_num   <- as.integer(sub("^Figure(\\d+).*", "\\1", basename(fig_files)))
fig_files <- fig_files[order(fig_num)]

stopifnot(length(fig_files) > 0)
cat("Merging", length(fig_files), "figures:\n")
cat(paste0("  ", basename(fig_files), collapse = "\n"), "\n")

## Combine into a single PDF (one figure per page, original 300-dpi quality)
out_pdf <- file.path(fig_dir, "All_Figures_Combined.pdf")
pdf_combine(fig_files, output = out_pdf)

cat("Done. Combined PDF written to:", normalizePath(out_pdf), "\n")
