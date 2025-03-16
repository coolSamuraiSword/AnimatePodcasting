#!/usr/bin/env python
"""
Test runner for AnimatePodcasting project.

This script runs all tests in the 'tests' directory and generates a report.
"""

import unittest
import os
import sys
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

def discover_and_run_tests():
    """Discover and run all tests in the 'tests' directory."""
    console.print("\n[bold cyan]Running AnimatePodcasting Tests[/bold cyan]\n")
    
    start_time = time.time()
    
    # Discover tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Prepare test results collector
    test_results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': []
    }
    
    # Run tests with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # Count tests for progress bar
        test_count = 0
        for suite in test_suite:
            for test_case in suite:
                test_count += test_case.countTestCases()
        
        task = progress.add_task("[bold blue]Running tests...", total=test_count)
        
        class ProgressTextTestResult(unittest.TextTestResult):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def startTest(self, test):
                super().startTest(test)
                test_name = test.id().split('.')[-1]
                progress.update(task, description=f"[bold blue]Running {test_name}...")
            
            def stopTest(self, test):
                super().stopTest(test)
                progress.update(task, advance=1)
            
            def addSuccess(self, test):
                super().addSuccess(test)
                test_results['passed'] += 1
                test_results['total'] += 1
                test_results['details'].append({
                    'name': test.id(),
                    'status': 'passed',
                    'message': None
                })
            
            def addFailure(self, test, err):
                super().addFailure(test, err)
                test_results['failed'] += 1
                test_results['total'] += 1
                test_results['details'].append({
                    'name': test.id(),
                    'status': 'failed',
                    'message': str(err[1])
                })
            
            def addError(self, test, err):
                super().addError(test, err)
                test_results['errors'] += 1
                test_results['total'] += 1
                test_results['details'].append({
                    'name': test.id(),
                    'status': 'error',
                    'message': str(err[1])
                })
        
        runner = unittest.TextTestRunner(resultclass=ProgressTextTestResult, verbosity=0)
        runner.run(test_suite)
    
    end_time = time.time()
    
    # Display results
    console.print("\n[bold cyan]Test Results[/bold cyan]")
    
    results_table = Table(show_header=True)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Total Tests", str(test_results['total']))
    results_table.add_row("Passed", f"[green]{test_results['passed']}[/green]")
    
    if test_results['failed'] > 0:
        results_table.add_row("Failed", f"[red]{test_results['failed']}[/red]")
    else:
        results_table.add_row("Failed", "0")
    
    if test_results['errors'] > 0:
        results_table.add_row("Errors", f"[red]{test_results['errors']}[/red]")
    else:
        results_table.add_row("Errors", "0")
    
    results_table.add_row("Time", f"{end_time - start_time:.2f} seconds")
    
    console.print(results_table)
    
    # Display failures and errors if any
    if test_results['failed'] > 0 or test_results['errors'] > 0:
        console.print("\n[bold red]Failures and Errors:[/bold red]")
        
        failures_table = Table(show_header=True)
        failures_table.add_column("Test", style="cyan")
        failures_table.add_column("Status", style="yellow")
        failures_table.add_column("Message", style="red")
        
        for detail in test_results['details']:
            if detail['status'] in ['failed', 'error']:
                status_color = "red" if detail['status'] == 'error' else "yellow"
                status = detail['status'].upper()
                failures_table.add_row(
                    detail['name'],
                    f"[{status_color}]{status}[/{status_color}]",
                    detail['message']
                )
        
        console.print(failures_table)
    
    # Return test results
    return test_results

if __name__ == "__main__":
    try:
        # Ensure the current directory is in the path
        sys.path.insert(0, os.path.abspath('.'))
        
        # Run tests
        results = discover_and_run_tests()
        
        # Set exit code based on test results
        if results['failed'] > 0 or results['errors'] > 0:
            sys.exit(1)
        else:
            console.print("\n[bold green]All tests passed![/bold green]")
            sys.exit(0)
            
    except Exception as e:
        console.print(f"\n[bold red]Error running tests: {str(e)}[/bold red]")
        sys.exit(1) 