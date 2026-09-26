# Verify the generated page and image that will actually be deployed.
# Usage: bundle exec ruby tools/check_swe2_social_card.rb [build directory]
require "nokogiri"
require "pathname"
require "uri"

site = Pathname.new(ARGV.fetch(0, "_site")).realpath
page = site.join("swe-2-extended/index.html")
abort "Missing built SWE-2 page: #{page}" unless page.file?
head = Nokogiri::HTML(page.read(encoding: "UTF-8"), nil, "UTF-8").at_css("head")

meta = lambda do |key|
  tags = head.css("meta[name='#{key}'], meta[property='#{key}']")
  abort "Expected one nonempty #{key} tag" unless tags.length == 1 && !tags.first["content"].to_s.empty?
  tags.first["content"]
end

abort "SWE-2 must use a large X card" unless meta.call("twitter:card") == "summary_large_image"
abort "Wrong canonical card URL" unless meta.call("og:url") == "https://anishlk.com/swe-2-extended/"
abort "X card title is too long" unless meta.call("twitter:title").length <= 70
abort "X card description is too long" unless meta.call("og:description").length <= 200
meta.call("twitter:site")
meta.call("twitter:image:alt")

image_url = meta.call("twitter:image")
abort "X and Open Graph must use the same cover" unless image_url == meta.call("og:image")
image_uri = URI(image_url)
unless image_uri.scheme == "https" && image_uri.host == "anishlk.com" && image_uri.query.nil?
  abort "Use an absolute HTTPS image URL with a versioned filename"
end
image = site.join(image_uri.path.delete_prefix("/")).cleanpath
abort "Cover is outside the build directory" unless image.to_s.start_with?("#{site}/")
abort "Missing social cover: #{image}" unless image.file?
abort "Social cover exceeds 5 MB" if image.size >= 5 * 1024 * 1024
abort "Social cover is not a JPEG" unless image.binread(3) == "\xFF\xD8\xFF".b
puts "SWE-2 large-image card verified: #{image_url} (#{image.size} bytes)"
