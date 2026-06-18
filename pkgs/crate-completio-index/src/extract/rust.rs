use std::path::Path;

use anyhow::Result;
use proc_macro2::{LineColumn, Span};
use syn::{
    ImplItem, ItemEnum, ItemFn, ItemImpl, ItemStruct, ItemTrait, ItemType, Type,
    visit::{self, Visit},
};

use super::Definition;
use crate::normalize::normalize_indentation;

pub fn extract(_path: &Path, source: &str) -> Result<Vec<Definition>> {
    let file = syn::parse_file(source)?;
    let mut visitor = RustVisitor {
        source,
        line_starts: line_starts(source),
        definitions: Vec::new(),
    };
    visitor.visit_file(&file);
    Ok(visitor.definitions)
}

struct RustVisitor<'a> {
    source: &'a str,
    line_starts: Vec<usize>,
    definitions: Vec<Definition>,
}

impl<'a> RustVisitor<'a> {
    fn push_definition(
        &mut self,
        kind: &str,
        name: String,
        parent_name: Option<String>,
        span: Span,
    ) {
        let Some((span_start, start_line, start_column)) =
            self.offset_from_line_column(span.start())
        else {
            return;
        };
        let Some((span_end, end_line, end_column)) = self.offset_from_line_column(span.end())
        else {
            return;
        };
        if span_start >= span_end || span_end > self.source.len() {
            return;
        }
        let code = self.source[span_start..span_end].to_string();
        let normalized_code = normalize_indentation(&code);
        if normalized_code.is_empty() {
            return;
        }

        self.definitions.push(Definition {
            kind: kind.to_string(),
            name,
            parent_name,
            span_start,
            span_end,
            start_line,
            start_column,
            end_line,
            end_column,
            code,
            normalized_code,
        });
    }

    fn offset_from_line_column(&self, lc: LineColumn) -> Option<(usize, usize, usize)> {
        if lc.line == 0 {
            return None;
        }
        let line_start = *self.line_starts.get(lc.line - 1)?;
        Some((line_start + lc.column, lc.line, lc.column + 1))
    }
}

impl<'ast> Visit<'ast> for RustVisitor<'_> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.push_definition(
            "rust:function",
            node.sig.ident.to_string(),
            None,
            node.span(),
        );
        visit::visit_item_fn(self, node);
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        self.push_definition("rust:struct", node.ident.to_string(), None, node.span());
        visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'ast ItemEnum) {
        self.push_definition("rust:enum", node.ident.to_string(), None, node.span());
        visit::visit_item_enum(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast ItemTrait) {
        self.push_definition("rust:trait", node.ident.to_string(), None, node.span());
        visit::visit_item_trait(self, node);
    }

    fn visit_item_type(&mut self, node: &'ast ItemType) {
        self.push_definition("rust:type", node.ident.to_string(), None, node.span());
        visit::visit_item_type(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        let parent_name = impl_parent_name(node);
        for item in &node.items {
            if let ImplItem::Fn(method) = item {
                self.push_definition(
                    "rust:method",
                    method.sig.ident.to_string(),
                    parent_name.clone(),
                    method.span(),
                );
            }
        }
        visit::visit_item_impl(self, node);
    }
}

fn impl_parent_name(item: &ItemImpl) -> Option<String> {
    match &*item.self_ty {
        Type::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        _ => None,
    }
}

fn line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (idx, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            starts.push(idx + 1);
        }
    }
    starts
}

trait SynSpanExt {
    fn span(&self) -> Span;
}

impl SynSpanExt for ItemFn {
    fn span(&self) -> Span {
        syn::spanned::Spanned::span(self)
    }
}

impl SynSpanExt for ItemStruct {
    fn span(&self) -> Span {
        syn::spanned::Spanned::span(self)
    }
}

impl SynSpanExt for ItemEnum {
    fn span(&self) -> Span {
        syn::spanned::Spanned::span(self)
    }
}

impl SynSpanExt for ItemTrait {
    fn span(&self) -> Span {
        syn::spanned::Spanned::span(self)
    }
}

impl SynSpanExt for ItemType {
    fn span(&self) -> Span {
        syn::spanned::Spanned::span(self)
    }
}

impl SynSpanExt for syn::ImplItemFn {
    fn span(&self) -> Span {
        syn::spanned::Spanned::span(self)
    }
}
