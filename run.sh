export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

bundle exec jekyll serve \
  --livereload \
  --port "${JEKYLL_PORT:-4000}" \
  --livereload-port "${JEKYLL_LIVERELOAD_PORT:-35729}"
