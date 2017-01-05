<template>
  <!-- Main content -->
  <section class='content'>
    <!-- Info boxes -->
     <!--
    YYY
    -->
    <div class='row'>

        <div class='col-xs-12'>
            <div class="alert alert-danger alert-dismissible" v-if="!complete">
                <button type="button" class="close" data-dismiss="alert" aria-hidden="true">×</button>
                <h4><i class="icon fa fa-ban"></i>Alert!</h4>
                This is a {{sampled}}/{{total}} or {{sampled_mb.toFixed(2)}}MB/{{total_mb.toFixed(2)}}MB subsample of the whole dataset.  To load more images,
run picpac-server with a larger value of --max-peek in MB.
            </div>
			<ul class="nav nav-tabs">
                <li v-bind:class="{'grid-item':true, active:cls.id==0}" height="300" width="300" v-for="cls in sample_sizes"><a class="dropdown-toggle" data-toggle="tab" v-bind:href="'#cat_'+cls.id">Class {{cls.id}}[{{cls.size}}]</a></li>
			</ul>
            <div class="tab-content">
                <div v-bind:class="{'tab-pane': true, in: cls.id==0, active: cls.id==0}" v-bind:id="'cat_'+cls.id" v-for="cls in sample_sizes">
                    <ul class="pagination" v-if="cls.pages > 1">

                        <li v-bind:id="'page_' + cls.id + '_' + page" v-bind:class="{'page-item':true, active:page==1}" height="300" width="300" v-for="page in cls.pages"><a href='#' class="page-link" v-on:click='load_page(cls.id, page, cls.size)'>{{page}}</a></li>
                    </ul>
                    <div v-bind:id="'samples_' + cls.id">
                        <img v-bind:src="'/api/thumb?class='+cls.id+'&offset=' + (off-1)" v-for="off in Math.min(50, cls.size)"/>
                    </div>
                        <!--
                        <div v-bind:class="{'tab-pane': true, in: page==1, active: page==1}" v-bind:id="'page'+page" v-for="page in (cls[1]+99)/100">
                            <img v-bind:src="'/api/thumb?class='+cls[0]+'&offset='+(xx-1)" v-for="xx in 100"></img>
                        </div>
                        -->

                </div> <!--pane-->
            </div> <!-- tab-content -->
		</div>
    </div> <!-- /.row -->
  </section>
  <!-- /.content -->
</template>

<script>
import $ from 'jquery'

function PageHtml (cls, page, size) {
  console.log('HTML:' + cls + '/' + page + '/' + size)
  var html = ''
  var begin = (page - 1) * 50
  var end = begin + 50
  if (end > size) {
    end = size
  }
  for (var i = begin; i < end; i++) {
    html += '<img src="/api/thumb?class=' + cls + '&offset=' + i + '"></img>'
  }
  return html
}

module.exports = {
  data: function () {
    return {
      sample_sizes: [],
      complete: true,
      sampled: 0,
      sampled_mb: 0,
      total: 0,
      total_mb: 0
    }
  },
  filters: {
    page_html: PageHtml
  },
  methods: {
    load_page: function (cls, page, size) {
      var par = '#cat_' + cls
      var pid = '#page_' + cls + '_' + page
      var sid = '#samples_' + cls
      $(par + ' li.active').removeClass('active')
      $(pid).addClass('active')
      console.log(pid)
      $(sid).html(PageHtml(cls, page, size))
    }
  },
  mounted: function () {
    var thisData = this
    this.$nextTick(function () {
      $.getJSON('/api/overview', function (overview) {
        console.log(overview.SAMPLE_SIZES)
        var xx = []
        for (var i in overview.SAMPLE_SIZES) {
          var yy = overview.SAMPLE_SIZES[i]
          xx.push({'id': i, 'pages': Math.ceil(yy / 50), 'size': yy})
        }
        thisData.sample_sizes = xx
        for (var j in overview.SUMMARY) {
          var f = overview.SUMMARY[j]
          if (f.key === 'ALL_SCANNED') {
            thisData.complete = f.value
            console.log('comp: ' + f.value)
          } else if (f.key === 'TOTAL_IMAGES') {
            thisData.total = f.value
          } else if (f.key === 'TOTAL_MB') {
            thisData.total_mb = f.value
          } else if (f.key === 'SCANNED_IMAGES') {
            thisData.sampled = f.value
          } else if (f.key === 'SCANNED_MB') {
            thisData.sampled_mb = f.value
          }
        }
      })
    })
  }
}
</script>
