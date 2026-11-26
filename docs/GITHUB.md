# Publishing to GitHub

The project files are prepared locally. To publish them, create an empty GitHub repository and run these commands from the ZoomRec directory. Replace `YOUR_ACCOUNT` and `YOUR_REPOSITORY` with the repository owner and name.

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Add ZoomRec leave-one-out implementation"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Use your configured Git credentials or an SSH remote. Keep tokens, passwords and private keys out of source control.

The default ignore rules cover data, weights, virtual environments, run outputs, logs and caches. Documentation links are relative to the project. If you configure other output directories, check `git status` before committing.

The release does not specify a software license, author list or paper citation. Choose a license and add a `LICENSE` file if you intend to grant reuse rights. The dataset and Qwen model remain subject to their respective providers' terms.
