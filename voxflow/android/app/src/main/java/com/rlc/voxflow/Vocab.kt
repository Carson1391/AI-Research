package com.rlc.voxflow

import android.content.res.AssetManager

/**
 * Post-ASR vocabulary repair, driven by assets/hotwords.txt.
 *
 * Line formats (same file as the desktop app):
 *   spoken form => written form     forced replacement, case-insensitive
 *   CanonicalTerm                   casing fix for exact word matches
 *   # comment
 */
class Vocab(assets: AssetManager) {

    private val rules = ArrayList<Pair<Regex, String>>()
    private val terms = ArrayList<Pair<Regex, String>>()

    init {
        try {
            assets.open("hotwords.txt").bufferedReader().forEachLine { raw ->
                val line = raw.trim()
                if (line.isEmpty() || line.startsWith("#")) return@forEachLine
                if (line.contains("=>")) {
                    val parts = line.split("=>", limit = 2)
                    val spoken = parts[0].trim()
                    val written = parts[1].trim()
                    if (spoken.isNotEmpty() && written.isNotEmpty()) {
                        rules.add(
                            Regex(
                                "\\b" + Regex.escape(spoken) + "\\b",
                                RegexOption.IGNORE_CASE
                            ) to written
                        )
                    }
                } else {
                    // canonical casing: "nemotron" -> "Nemotron"
                    terms.add(
                        Regex(
                            "\\b" + Regex.escape(line) + "\\b",
                            RegexOption.IGNORE_CASE
                        ) to line
                    )
                }
            }
        } catch (_: Exception) {
            // no vocab file bundled; repair becomes a no-op
        }
    }

    fun repair(input: String): String {
        var text = input
        for ((pattern, written) in rules) {
            text = pattern.replace(text, Regex.escapeReplacement(written))
        }
        for ((pattern, canonical) in terms) {
            text = pattern.replace(text, Regex.escapeReplacement(canonical))
        }
        return text
    }
}
