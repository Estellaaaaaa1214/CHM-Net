# Push Instructions

The anonymous.4open.science URL is a review/display link, not a Git push URL.
Push this prepared repository to the real GitHub repository that is connected to
the anonymous 4open.science mirror.

```bash
cd chm_net_anonymous_release
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```

After pushing, refresh the anonymous 4open.science project or reconnect it to the
same GitHub repository from the 4open.science dashboard.
