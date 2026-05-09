# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll", "~> 4.3"

gem "html-proofer", "~> 5.0", group: :test

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

group :jekyll_plugins do
  gem 'jekyll-compose'
  gem 'jekyll-sitemap'
  gem 'jekyll-seo-tag'
  gem 'jekyll-paginate'
  gem 'jekyll-redirect-from'
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]
