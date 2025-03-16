# Contributing to AnimatePodcasting

Thank you for considering contributing to AnimatePodcasting!

## Git Workflow

We follow a simplified Git workflow:

1. **Main Branch**: The `main` branch contains stable code.
2. **Feature Branches**: Create feature branches for new work.
3. **Pull Requests**: Submit PRs for code review before merging.

### Development Workflow

1. Update your local main branch:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Create a new feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make changes and commit frequently:
   ```bash
   git add -A
   git commit -m "Meaningful commit message describing the change"
   ```

4. Push your branch to the remote repository:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request on GitHub for code review.

## Commit Message Guidelines

Write clear, concise commit messages:

- Use present tense ("Add feature" not "Added feature")
- Start with a verb
- Keep the first line under 50 characters
- Reference issues or tickets when relevant

Examples:
- "Add caching system for transcriptions"
- "Fix bug in image generation with large prompts"
- "Improve CLI progress display"

## Code Style

- Follow PEP 8 style guidelines
- Use docstrings for all functions and classes
- Keep functions focused on a single responsibility
- Add type hints where possible

## Testing

- Write tests for new functionality
- Ensure all existing tests pass before submitting PR
- Consider edge cases in your tests 