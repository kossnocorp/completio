use std::path::Path;

use anyhow::{Result, anyhow};
use oxc_allocator::Allocator;
use oxc_ast::ast::{
    ArrowFunctionExpression, Class, Function, MethodDefinition, TSEnumDeclaration,
    TSInterfaceDeclaration, TSTypeAliasDeclaration, VariableDeclarator,
};
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_span::SourceType;
use oxc_syntax::scope::ScopeFlags;

use super::Definition;
use crate::normalize::normalize_indentation;

pub fn extract(path: &Path, source: &str) -> Result<Vec<Definition>> {
    let allocator = Allocator::new();
    let source_type = SourceType::from_path(path).map_err(|err| anyhow!(err.to_string()))?;
    let parsed = Parser::new(&allocator, source, source_type).parse();

    let mut visitor = TsVisitor {
        source,
        definitions: Vec::new(),
        class_stack: Vec::new(),
    };
    visitor.visit_program(&parsed.program);
    Ok(visitor.definitions)
}

struct TsVisitor<'a> {
    source: &'a str,
    definitions: Vec<Definition>,
    class_stack: Vec<String>,
}

impl<'a> TsVisitor<'a> {
    fn push_definition(
        &mut self,
        kind: &str,
        name: String,
        parent_name: Option<String>,
        span_start: usize,
        span_end: usize,
    ) {
        if span_start >= span_end || span_end > self.source.len() {
            return;
        }

        let code = self.source[span_start..span_end].to_string();
        let normalized_code = normalize_indentation(&code);
        if normalized_code.is_empty() {
            return;
        }

        let (start_line, start_column) = line_col(self.source, span_start);
        let (end_line, end_column) = line_col(self.source, span_end);

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
}

impl<'ast> Visit<'ast> for TsVisitor<'_> {
    fn visit_function(&mut self, node: &Function<'ast>, flags: ScopeFlags) {
        if let Some(name) = node.name() {
            self.push_definition(
                "ts:function",
                name.to_string(),
                None,
                node.span.start as usize,
                node.span.end as usize,
            );
        }
        walk::walk_function(self, node, flags);
    }

    fn visit_variable_declarator(&mut self, node: &VariableDeclarator<'ast>) {
        let Some(binding_name) = node.id.get_identifier_name().map(|name| name.to_string()) else {
            walk::walk_variable_declarator(self, node);
            return;
        };

        if let Some(init) = node.init.as_ref() {
            match init {
                oxc_ast::ast::Expression::ArrowFunctionExpression(arrow) => {
                    record_arrow(self, &binding_name, arrow);
                }
                oxc_ast::ast::Expression::FunctionExpression(function) => {
                    self.push_definition(
                        "ts:function",
                        binding_name.clone(),
                        None,
                        function.span.start as usize,
                        function.span.end as usize,
                    );
                }
                _ => {}
            }
        }

        walk::walk_variable_declarator(self, node);
    }

    fn visit_class(&mut self, node: &Class<'ast>) {
        let class_name = node.name().map(|name| name.to_string());
        if let Some(class_name) = class_name {
            self.push_definition(
                "ts:class",
                class_name.clone(),
                None,
                node.span.start as usize,
                node.span.end as usize,
            );
            self.class_stack.push(class_name);
            walk::walk_class(self, node);
            self.class_stack.pop();
        } else {
            walk::walk_class(self, node);
        }
    }

    fn visit_method_definition(&mut self, node: &MethodDefinition<'ast>) {
        let name = node
            .key
            .name()
            .map(|name| name.to_string())
            .unwrap_or_else(|| "<computed>".to_string());
        let parent_name = self.class_stack.last().cloned();
        self.push_definition(
            "ts:method",
            name,
            parent_name,
            node.span.start as usize,
            node.span.end as usize,
        );
        walk::walk_method_definition(self, node);
    }

    fn visit_ts_type_alias_declaration(&mut self, node: &TSTypeAliasDeclaration<'ast>) {
        self.push_definition(
            "ts:type",
            node.id.name.to_string(),
            None,
            node.span.start as usize,
            node.span.end as usize,
        );
        walk::walk_ts_type_alias_declaration(self, node);
    }

    fn visit_ts_interface_declaration(&mut self, node: &TSInterfaceDeclaration<'ast>) {
        self.push_definition(
            "ts:interface",
            node.id.name.to_string(),
            None,
            node.span.start as usize,
            node.span.end as usize,
        );
        walk::walk_ts_interface_declaration(self, node);
    }

    fn visit_ts_enum_declaration(&mut self, node: &TSEnumDeclaration<'ast>) {
        self.push_definition(
            "ts:enum",
            node.id.name.to_string(),
            None,
            node.span.start as usize,
            node.span.end as usize,
        );
        walk::walk_ts_enum_declaration(self, node);
    }
}

fn record_arrow(
    visitor: &mut TsVisitor<'_>,
    binding_name: &str,
    arrow: &ArrowFunctionExpression<'_>,
) {
    visitor.push_definition(
        "ts:function",
        binding_name.to_string(),
        None,
        arrow.span.start as usize,
        arrow.span.end as usize,
    );
}

fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut column = 1;
    for (idx, ch) in source.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}
