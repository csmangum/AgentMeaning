#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the pipeline module.

This file contains tests for:
1. Base pipeline functionality
2. Individual component tests
3. Pipeline composition
4. Custom components
"""

import unittest
import torch
import numpy as np
from typing import Dict, Tuple, Any

from meaning_transform.src.pipeline import (
    Pipeline,
    PipelineComponent,
    CustomComponent,
    BranchComponent,
    ConditionalComponent,
    PipelineFactory
)


class TestPipelineComponent(PipelineComponent):
    """Test pipeline component that remembers if it was called."""
    
    def __init__(self, return_value=None, name=None):
        self.called = False
        self.return_value = return_value
        self._name = name
        
    @property
    def name(self) -> str:
        return self._name or "TestComponent"
        
    def process(self, data, context=None):
        if context is None:
            context = {}
            
        self.called = True
        context['test_component_called'] = True
        
        if self.return_value is not None:
            return self.return_value, context
        else:
            return data, context


class TestDataTransformComponent(PipelineComponent):
    """Test component that transforms data."""
    
    def __init__(self, transform_func, name=None):
        self.transform_func = transform_func
        self._name = name
        
    @property
    def name(self) -> str:
        return self._name or "TransformComponent"
        
    def process(self, data, context=None):
        if context is None:
            context = {}
            
        transformed = self.transform_func(data)
        context['transformed'] = True
        
        return transformed, context


class PipelineTest(unittest.TestCase):
    """Test case for the Pipeline class."""
    
    def test_empty_pipeline(self):
        """Test an empty pipeline."""
        pipeline = Pipeline()
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertEqual(result, data)
        self.assertEqual(context['input'], data)
        self.assertEqual(context['output'], data)
        
    def test_single_component(self):
        """Test a pipeline with a single component."""
        component = TestPipelineComponent()
        pipeline = Pipeline().add(component)
        
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertTrue(component.called)
        self.assertTrue(context['test_component_called'])
        self.assertEqual(result, data)
        
    def test_multiple_components(self):
        """Test a pipeline with multiple components."""
        components = [TestPipelineComponent() for _ in range(3)]
        pipeline = Pipeline().add_many(components)
        
        data = "test"
        result, context = pipeline.process(data)
        
        for component in components:
            self.assertTrue(component.called)
        self.assertTrue(context['test_component_called'])
        self.assertEqual(result, data)
        
    def test_component_transformation(self):
        """Test that components can transform data."""
        transform_func = lambda x: x.upper()
        component = TestDataTransformComponent(transform_func)
        pipeline = Pipeline().add(component)
        
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertEqual(result, "TEST")
        self.assertTrue(context['transformed'])
        
    def test_pipeline_chaining(self):
        """Test that pipelines can be chained together."""
        transform1 = TestDataTransformComponent(lambda x: x.upper())
        transform2 = TestDataTransformComponent(lambda x: x + "!")
        pipeline = Pipeline().add(transform1).add(transform2)
        
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertEqual(result, "TEST!")
        self.assertTrue(context['transformed'])
        
    def test_pipeline_context_passing(self):
        """Test that context is properly passed between components."""
        def add_to_context(data, context=None):
            if context is None:
                context = {}
            context['key1'] = 'value1'
            return data, context
            
        def check_context(data, context=None):
            if context is None:
                context = {}
            context['key2'] = 'value2'
            context['found_key1'] = 'key1' in context
            return data, context
            
        pipeline = Pipeline().add_many([
            CustomComponent(add_to_context, name="AddContext"),
            CustomComponent(check_context, name="CheckContext")
        ])
        
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertEqual(result, data)
        self.assertEqual(context['key1'], 'value1')
        self.assertEqual(context['key2'], 'value2')
        self.assertTrue(context['found_key1'])
        
    def test_conditional_component(self):
        """Test conditional components."""
        component = TestPipelineComponent()
        
        # Create predicates
        true_predicate = lambda data, ctx: True
        false_predicate = lambda data, ctx: False
        
        # Create conditional components
        true_conditional = ConditionalComponent(true_predicate, component)
        false_conditional = ConditionalComponent(false_predicate, component)
        
        # Reset called flag
        component.called = False
        
        # Test true conditional
        pipeline = Pipeline().add(true_conditional)
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertTrue(component.called)
        
        # Reset called flag
        component.called = False
        
        # Test false conditional
        pipeline = Pipeline().add(false_conditional)
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertFalse(component.called)
        
    def test_branch_component(self):
        """Test branch component."""
        # Components for different branches
        upper_component = TestDataTransformComponent(lambda x: x.upper(), name="Upper")
        lower_component = TestDataTransformComponent(lambda x: x.lower(), name="Lower")
        reverse_component = TestDataTransformComponent(lambda x: x[::-1], name="Reverse")
        
        # Create branch component
        branch = BranchComponent({
            'upper': upper_component,
            'lower': lower_component,
            'reverse': reverse_component
        })
        
        # Create pipeline
        pipeline = Pipeline().add(branch)
        
        # Process data
        data = "Test"
        result, context = pipeline.process(data)
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertEqual(result['upper'], "TEST")
        self.assertEqual(result['lower'], "test")
        self.assertEqual(result['reverse'], "tseT")
        self.assertTrue('branch_results' in context)
        self.assertTrue('branch_contexts' in context)
        
    def test_pipeline_replace(self):
        """Test replacing a component in the pipeline."""
        transform1 = TestDataTransformComponent(lambda x: x.upper())
        transform2 = TestDataTransformComponent(lambda x: x.lower())
        
        pipeline = Pipeline().add(transform1)
        
        # Replace the component
        pipeline.replace(0, transform2)
        
        data = "Test"
        result, context = pipeline.process(data)
        
        self.assertEqual(result, "test")
        
    def test_pipeline_insert(self):
        """Test inserting a component in the pipeline."""
        transform1 = TestDataTransformComponent(lambda x: x.upper())
        transform2 = TestDataTransformComponent(lambda x: x + "!")
        
        pipeline = Pipeline().add(transform1)
        
        # Insert a component
        pipeline.insert(0, transform2)
        
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertEqual(result, "TEST!!")
        
    def test_pipeline_remove(self):
        """Test removing a component from the pipeline."""
        transform1 = TestDataTransformComponent(lambda x: x.upper())
        transform2 = TestDataTransformComponent(lambda x: x + "!")
        
        pipeline = Pipeline().add_many([transform1, transform2])
        
        # Remove the second component
        pipeline.remove(1)
        
        data = "test"
        result, context = pipeline.process(data)
        
        self.assertEqual(result, "TEST")


if __name__ == '__main__':
    unittest.main() 