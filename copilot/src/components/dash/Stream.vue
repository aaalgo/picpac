<template>
  <!-- Main content -->
  <section class='content'>
      <div class='row'>
             <div class="col-xs-12">
          <div class="grid">
              <div class="grid-item" height="300" width="300" v-for="image in images">
                  <img v-bind:src="image"></img>
              </div>
          </div>
      </div>
      </div>
  </section>
  <!-- /.content -->
</template>

<script>
import $ from 'jquery'
import Masonry from 'masonry-layout'
import imagesLoaded from 'imagesloaded'

module.exports = {
  data: function () {
    return {
      images: []
    }
  },
  mounted: function () {
    var thisData = this
    this.$nextTick(function () {
      $.getJSON('/api/sample', function (data) {
        var x = []
        for (var i in data.samples) {
          var sid = data.samples[i]
          x.push('/api/image?max_size=300&id=' + sid)
        }
        thisData.images = x
        thisData.$nextTick(function () {
          imagesLoaded('.grid', function () {
            var msnry = new Masonry('.grid', {
              itemSelector: '.grid-item',
              columnWidth: 200
            })
            console.log(msnry)
            msnry.on('layoutComplete', function (a, b) {
              console.log('layout done')
              console.log(b)
            })
            msnry.layout()
          })
        })
      })
    })
  }
}
</script>
<style>
.info-box {
  cursor: pointer;
}
.info-box-content {
  text-align: center;
  vertical-align: middle;
  display: inherit;
}
.fullCanvas {
  width: 100%;
}
</style>
