$("#like_button").click(function() {
  $.get('/likes/<int:id>')
})
