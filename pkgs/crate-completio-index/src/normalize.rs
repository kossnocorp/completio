use xxhash_rust::xxh3::xxh3_128;

pub fn normalize_indentation(input: &str) -> String {
    let lines: Vec<&str> = input.lines().collect();
    let start = lines.iter().position(|line| !line.trim().is_empty());
    let end = lines.iter().rposition(|line| !line.trim().is_empty());

    let Some(start) = start else {
        return String::new();
    };
    let end = end.expect("end exists when start exists");
    let body = &lines[start..=end];

    let min_indent = body
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.chars()
                .take_while(|ch| matches!(ch, ' ' | '\t'))
                .count()
        })
        .min()
        .unwrap_or(0);

    body.iter()
        .map(|line| {
            let trimmed = if line.len() >= min_indent {
                &line[min_indent..]
            } else {
                line.trim_start_matches([' ', '\t'])
            };
            trimmed.trim_end()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn hash_hex(input: impl AsRef<[u8]>) -> String {
    format!("{:032x}", xxh3_128(input.as_ref()))
}

pub fn embedding_text(
    path: &str,
    kind: &str,
    name: &str,
    parent: Option<&str>,
    code: &str,
) -> String {
    let mut out = String::new();
    out.push_str("path: ");
    out.push_str(path);
    out.push_str("\nkind: ");
    out.push_str(kind);
    out.push_str("\nname: ");
    out.push_str(name);
    if let Some(parent) = parent {
        out.push_str("\nparent: ");
        out.push_str(parent);
    }
    out.push_str("\ncode:\n");
    out.push_str(code);
    out
}
